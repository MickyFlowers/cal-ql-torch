"""
ACT (Action Chunking with Transformers) Model Implementation

Based on "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
https://tonyzhaozh.github.io/aloha/

Key components:
- CVAE Encoder: Encodes observation and action sequence into latent z
- Transformer Decoder: Decodes observation + z into action chunk
- Vision Backbone: ResNet18 for image feature extraction
"""

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoidal position encoding table."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class VisionBackbone(nn.Module):
    """Vision encoder using ResNet18 backbone."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 512,
        train_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features[-1].shape[1]

        # Project to hidden_dim
        self.proj = nn.Conv2d(self.feature_dim, hidden_dim, kernel_size=1)

        # Freeze backbone if not training
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Args:
            images: (B, C, H, W) or (B, N_cam, C, H, W)
        Returns:
            features: (B, N_tokens, hidden_dim)
        """
        if images.dim() == 5:
            # Multiple cameras: (B, N_cam, C, H, W)
            B, N_cam, C, H, W = images.shape
            images = images.view(B * N_cam, C, H, W)
            features = self.backbone(images)[-1]  # (B*N_cam, feat_dim, h, w)
            features = self.proj(features)  # (B*N_cam, hidden_dim, h, w)
            _, D, h, w = features.shape
            features = features.view(B, N_cam, D, h, w)
            features = features.permute(0, 1, 3, 4, 2)  # (B, N_cam, h, w, D)
            features = features.reshape(B, N_cam * h * w, D)  # (B, N_tokens, D)
        else:
            # Single camera: (B, C, H, W)
            features = self.backbone(images)[-1]  # (B, feat_dim, h, w)
            features = self.proj(features)  # (B, hidden_dim, h, w)
            B, D, h, w = features.shape
            features = features.permute(0, 2, 3, 1)  # (B, h, w, D)
            features = features.reshape(B, h * w, D)  # (B, N_tokens, D)

        return features


class CVAEEncoder(nn.Module):
    """
    CVAE Encoder: Encodes action sequence into latent z.

    Uses a transformer to process [CLS] + proprio + actions,
    then outputs mean and variance for the latent distribution.
    """

    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        chunk_size: int = 100,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size

        # Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.proprio_embed = nn.Linear(proprio_dim, hidden_dim)

        # Position encoding for action sequence
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=chunk_size + 2)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads for mean and log_var
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Initialize
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, proprio, actions):
        """
        Args:
            proprio: (B, proprio_dim) proprioceptive state
            actions: (B, chunk_size, action_dim) action sequence
        Returns:
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log variance of latent distribution
        """
        B = proprio.shape[0]

        # Embed inputs
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        proprio_embed = self.proprio_embed(proprio).unsqueeze(1)  # (B, 1, D)
        action_embed = self.action_embed(actions)  # (B, chunk_size, D)

        # Concatenate: [CLS, proprio, actions]
        x = torch.cat([cls_tokens, proprio_embed, action_embed], dim=1)  # (B, 2+chunk_size, D)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Use CLS token output for latent distribution
        cls_output = x[:, 0]  # (B, D)
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)

        return mu, logvar


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder: Generates action chunk from observation and latent z.

    Uses cross-attention between learned action queries and
    encoded observation features.
    """

    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 8,
        chunk_size: int = 100,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Input projections
        self.proprio_embed = nn.Linear(proprio_dim, hidden_dim)
        self.latent_embed = nn.Linear(latent_dim, hidden_dim)

        # Learnable action queries
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        nn.init.normal_(self.action_queries, std=0.02)

        # Positional encoding for queries
        self.query_pos = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
        nn.init.normal_(self.query_pos, std=0.02)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to action space
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, vision_features, proprio, z):
        """
        Args:
            vision_features: (B, N_tokens, hidden_dim) from vision backbone
            proprio: (B, proprio_dim) proprioceptive state
            z: (B, latent_dim) latent variable
        Returns:
            actions: (B, chunk_size, action_dim) predicted action chunk
        """
        B = proprio.shape[0]

        # Embed proprio and latent
        proprio_embed = self.proprio_embed(proprio).unsqueeze(1)  # (B, 1, D)
        latent_embed = self.latent_embed(z).unsqueeze(1)  # (B, 1, D)

        # Build memory for cross-attention: [vision, proprio, latent]
        memory = torch.cat([vision_features, proprio_embed, latent_embed], dim=1)

        # Prepare action queries
        queries = self.action_queries.expand(B, -1, -1) + self.query_pos

        # Decode
        output = self.transformer_decoder(queries, memory)

        # Project to action space
        actions = self.action_head(output)

        return actions


class ACTPolicy(nn.Module):
    """
    ACT (Action Chunking with Transformers) Policy.

    A CVAE-based policy that predicts action chunks using:
    - Vision backbone for image features
    - CVAE encoder for learning latent representation during training
    - Transformer decoder for action prediction

    During training: samples z from posterior q(z|o,a)
    During inference: uses z = 0 (prior mean)
    """

    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        chunk_size: int = 100,
        dim_feedforward: int = 2048,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = True,
        train_backbone: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

        # Vision backbone
        self.vision_backbone = VisionBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            hidden_dim=hidden_dim,
            train_backbone=train_backbone,
        )

        # CVAE encoder (only used during training)
        self.cvae_encoder = CVAEEncoder(
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            chunk_size=chunk_size,
        )

        # Transformer decoder
        self.decoder = TransformerDecoder(
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            chunk_size=chunk_size,
            dim_feedforward=dim_feedforward,
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, proprio, images, actions=None, deterministic=False):
        """
        Forward pass for training or inference.

        Args:
            proprio: (B, proprio_dim) proprioceptive state
            images: (B, C, H, W) or (B, N_cam, C, H, W) camera images
            actions: (B, chunk_size, action_dim) ground truth actions (training only)
            deterministic: if True, use z=0 for inference

        Returns:
            pred_actions: (B, chunk_size, action_dim) predicted actions
            mu: (B, latent_dim) mean (None if deterministic)
            logvar: (B, latent_dim) log variance (None if deterministic)
        """
        B = proprio.shape[0]

        # Extract vision features
        vision_features = self.vision_backbone(images)

        if actions is not None and not deterministic:
            # Training mode: encode actions to get posterior
            mu, logvar = self.cvae_encoder(proprio, actions)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference mode: use prior mean z = 0
            z = torch.zeros(B, self.latent_dim, device=proprio.device)
            mu, logvar = None, None

        # Decode to get action chunk
        pred_actions = self.decoder(vision_features, proprio, z)

        return pred_actions, mu, logvar

    @torch.no_grad()
    def get_action(self, proprio, images, temporal_ensemble=None):
        """
        Get action for deployment.

        Args:
            proprio: (1, proprio_dim) or (proprio_dim,) proprioceptive state
            images: camera images
            temporal_ensemble: optional TemporalEnsemble object

        Returns:
            action: (action_dim,) single action to execute
        """
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() == 4 and images.shape[0] != 1:
            # Multiple cameras without batch dim
            images = images.unsqueeze(0)

        pred_actions, _, _ = self.forward(proprio, images, deterministic=True)

        if temporal_ensemble is not None:
            action = temporal_ensemble.update(pred_actions[0])
        else:
            action = pred_actions[0]  # First action of the chunk

        return action

    def freeze_backbone(self):
        """Freeze vision backbone parameters."""
        for param in self.vision_backbone.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze vision backbone parameters."""
        for param in self.vision_backbone.backbone.parameters():
            param.requires_grad = True


class TemporalEnsemble:
    """
    Temporal ensemble for smoother action execution.

    Combines predictions from overlapping action chunks using
    exponential weighting.
    """

    def __init__(self, chunk_size: int, action_dim: int, decay: float = 0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay = decay

        # Pre-compute weights
        self.weights = torch.exp(-decay * torch.arange(chunk_size))
        self.weights = self.weights / self.weights.sum()

        # Buffer for storing predictions
        self.buffer = None
        self.timestep = 0

    def reset(self):
        """Reset the ensemble buffer."""
        self.buffer = None
        self.timestep = 0

    def update(self, action_chunk):
        """
        Update buffer with new action chunk and return ensembled action.

        Args:
            action_chunk: (chunk_size, action_dim) new predictions

        Returns:
            action: (action_dim,) ensembled action for current timestep
        """
        device = action_chunk.device

        if self.buffer is None:
            # Initialize buffer
            self.buffer = torch.zeros(
                self.chunk_size, self.chunk_size, self.action_dim, device=device
            )
            self.weights = self.weights.to(device)

        # Shift buffer
        if self.timestep > 0:
            self.buffer = torch.roll(self.buffer, -1, dims=0)
            self.buffer[-1] = 0

        # Add new predictions to buffer
        # action_chunk[i] is the prediction for timestep (current + i)
        for i in range(self.chunk_size):
            if self.timestep + i < self.chunk_size:
                continue
            idx = min(i, self.chunk_size - 1)
            self.buffer[idx, self.timestep % self.chunk_size] = action_chunk[i]

        # For simplicity, just use the first action during warmup
        if self.timestep < self.chunk_size:
            self.timestep += 1
            return action_chunk[0]

        # Compute weighted average
        valid_preds = self.buffer[0]  # Predictions for current timestep
        action = (valid_preds * self.weights.unsqueeze(-1)).sum(dim=0)

        self.timestep += 1
        return action
