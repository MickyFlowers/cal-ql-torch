import copy

import torch
from diffusers.optimization import get_scheduler

from model.ema_model import EMAModel


class Trainer:
    def __init__(self, policy, vision_encoder, config):
        self.policy = policy
        self.vision_encoder = vision_encoder
        params_to_optimize = list(self.policy.parameters()) + list(self.vision_encoder.parameters())
        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=config.learning_rate, betas=tuple(config.betas), weight_decay=config.weight_decay, eps=config.adam_epsilon)
        self.lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_epochs,
            num_cycles=config.num_cycles,
            power=config.lr_power,
        )
        self.policy_ema_model = copy.deepcopy(self.policy)
        self.vision_encoder_ema_model = copy.deepcopy(self.vision_encoder)
        self.policy_ema = EMAModel(
            self.policy_ema_model,
            update_after_step=config.policy_ema.update_after_step,
            inv_gamma=config.policy_ema.inv_gamma,
            power=config.policy_ema.power,
            min_value=config.policy_ema.min_value,
            max_value=config.policy_ema.max_value
        )
        self.vision_encoder_ema = EMAModel(
            self.vision_encoder_ema_model,
            update_after_step=config.vision_encoder_ema.update_after_step,
            inv_gamma=config.vision_encoder_ema.inv_gamma,
            power=config.vision_encoder_ema.power,
            min_value=config.vision_encoder_ema.min_value,
            max_value=config.vision_encoder_ema.max_value
        )
        
    
    def train_step(self, batch, eval=False):
        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
        # print(images.shape)
        # print(observations.shape)
        # print(actions.shape)
        actions = batch["action"].to(self.device)
        image_embeds = self.vision_encoder(images)[1]  # Use patch tokens
        loss = self.policy(
            img_tokens=image_embeds,
            state_tokens=observations,
            action_gt=actions,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.policy_ema.step(self.policy)
        self.vision_encoder_ema.step(self.vision_encoder)

        metrics = {
            "loss": loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }
        if eval:
            with torch.no_grad():
                sample_action = self.policy.predict_action(image_embeds, observations)
                action_error = torch.nn.functional.mse_loss(sample_action, actions)
                metrics.update({"action_error": action_error.item()})
        return metrics
    def save_checkpoint(self, filepath):
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "vision_encoder_state_dict": self.vision_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "policy_ema_state_dict": self.policy_ema_model.state_dict(),
            "vision_encoder_ema_state_dict": self.vision_encoder_ema_model.state_dict(),
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location="cpu")
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.vision_encoder.load_state_dict(checkpoint["vision_encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.policy_ema_model.load_state_dict(checkpoint["policy_ema_state_dict"])
        self.vision_encoder_ema_model.load_state_dict(checkpoint["vision_encoder_ema_state_dict"])
        print(f"Loaded checkpoint from {filepath}")

    def to_device(self, device):
        self.device = device
        self.policy.to(device)
        self.vision_encoder.to(device)
        self.policy_ema_model.to(device)
        self.vision_encoder_ema_model.to(device)