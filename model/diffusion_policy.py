import math
import re
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler
from timm.models.vision_transformer import (Attention, Mlp, RmsNorm,
                                            use_fused_attn)
from torch import nn
from torch.jit import Final


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask:  Union[torch.Tensor, None]) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x
    
class TransformerBlock(nn.Module):
    """
    A Transformer block with cross-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,**block_kwargs)
        
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, 
            act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, mask=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x
        
        origin_x = x
        x = self.norm2(x)
        x = self.cross_attn(x, c, mask)
        x = x + origin_x
                
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        
        return x
    
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, 
                                  embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim
    
    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)
    
    return c_pos_emb

class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x
    
class DiffusionModel(nn.Module):
    def __init__(
        self,
        output_dim=128,
        horizon=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        img_cond_len=4096,
        img_pos_embed_config=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.img_pos_embed_config = img_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon+2, hidden_size))
        # Image conditions
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len, hidden_size))

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                ('state', 1),
                ('action', self.horizon),
            ])
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep  embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, t, img_c, img_mask=None):
        
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, x], dim=1)               # (B, T+1, D)

        # Add multimodal position embeddings
        x = x + self.x_pos_embed

        img_c = img_c + self.img_cond_pos_embed
        # Forward pass
        c, mask = img_c, img_mask
        for _, block in enumerate(self.blocks):
            x = block(x, c, mask)                      
        x = self.final_layer(x)                     
        x = x[:, -self.horizon:]
        return x


class DiffusionPolicy(nn.Module):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 img_token_dim, state_token_dim, 
                 img_cond_len, img_pos_embed_config=None, dtype=torch.bfloat16):
        super().__init__()
        # Create diffusion model
        hidden_size = config.hidden_size
        self.model = DiffusionModel(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            img_cond_len=img_cond_len,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # Create adpators for various conditional inputs
        self.img_adaptor = self.build_condition_adapter(
            config.img_adaptor, 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config.state_adaptor, 
            in_features=state_token_dim,
            out_features=hidden_size
        )
        self.action_adaptor = self.build_condition_adapter(
            config.action_adaptor, 
            in_features=action_dim, 
            out_features=hidden_size
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config.noise_scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config.num_train_timesteps,
            beta_schedule=noise_scheduler_config.beta_schedule,
            prediction_type=noise_scheduler_config.prediction_type,
            clip_sample=noise_scheduler_config.clip_sample,
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config.num_train_timesteps,
            beta_schedule=noise_scheduler_config.beta_schedule,
            prediction_type=noise_scheduler_config.prediction_type,
        )

        self.num_train_timesteps = noise_scheduler_config.num_train_timesteps
        self.num_inference_timesteps = noise_scheduler_config.num_inference_timesteps
        self.prediction_type = noise_scheduler_config.prediction_type

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad] + 
            [p.numel() for p in self.img_adaptor.parameters() if p.requires_grad] + 
            [p.numel() for p in self.state_adaptor.parameters() if p.requires_grad] +
            [p.numel() for p in self.action_adaptor.parameters() if p.requires_grad]))
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, img_tokens, state_tokens, action_tokens=None):

        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)
        if action_tokens is not None:
            adpated_action = self.action_adaptor(action_tokens)
            return adpated_img, adpated_state, adpated_action
        return adpated_img, adpated_state

    def conditional_sample(self, img_cond, state_traj):
        
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim), 
            dtype=dtype, device=device)
        
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = noisy_action
            action_traj = self.action_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.model(state_action_traj,
                                    t.unsqueeze(-1).to(device),
                                    img_cond)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action

        return noisy_action
    
    # ========= Train  ============
    def compute_loss(self, img_tokens, state_tokens, action_gt) -> torch.Tensor:
        state_tokens = state_tokens.unsqueeze(1)
        batch_size = img_tokens.shape[0]
        device = img_tokens.device  

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # Align the dimension with the hidden size
        img_cond, state_traj, action_traj = self.adapt_conditions(
            img_tokens, state_tokens, noisy_action)
        
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)
        # Predict the denoised result
        pred = self.model(state_action_traj, timesteps, img_cond)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss
    
    # ========= Inference  ============
    def predict_action(self, img_tokens, state_tokens):
        '''
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
    
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = state_tokens.unsqueeze(1)
        img_cond, state_traj = self.adapt_conditions(img_tokens, state_tokens)
        
        # Run sampling
        action_pred = self.conditional_sample(img_cond, state_traj)
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)