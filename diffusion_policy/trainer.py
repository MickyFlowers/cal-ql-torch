import torch
from diffusers.optimization import get_scheduler


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
            power=config.power,
        )
        
    
    def train_step(self, batch, eval=False):
        observations = batch["observations"]['proprio'].to(self.device)
        images = batch["observations"]['image'].to(self.device)
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
        
    def to_device(self, device):
        self.device = device
        self.policy.to(device)
        self.vision_encoder.to(device)