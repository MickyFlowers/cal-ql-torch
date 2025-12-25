"""
ACT Training Script

Train ACT (Action Chunking with Transformers) policy using imitation learning.
"""

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from act.act_model import ACTPolicy
from act.act_trainer import ACTTrainer
from data.dataset import ACTDataset
from utils.logger import WandBLogger


@hydra.main(config_path="../config", config_name="train_act", version_base=None)
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logging
    variant = OmegaConf.to_container(cfg, resolve=True)
    wandb_logger = WandBLogger(config=cfg.logging, variant=variant)

    # Create dataset - use config object directly like other datasets
    print(f"Loading dataset from {cfg.dataset.root_path}")
    dataset = ACTDataset(cfg.dataset)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"Action dim: {dataset.action_dim}")
    print(f"Proprio dim: {dataset.proprio_dim}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Create model
    policy = ACTPolicy(
        action_dim=dataset.action_dim,
        proprio_dim=dataset.proprio_dim,
        hidden_dim=cfg.act.hidden_dim,
        latent_dim=cfg.act.latent_dim,
        num_encoder_layers=cfg.act.num_encoder_layers,
        num_decoder_layers=cfg.act.num_decoder_layers,
        num_heads=cfg.act.num_heads,
        chunk_size=cfg.act.chunk_size,
        dim_feedforward=cfg.act.dim_feedforward,
        backbone_name=cfg.act.backbone_name,
        pretrained_backbone=cfg.act.pretrained_backbone,
        train_backbone=cfg.act.train_backbone,
    )

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = ACTTrainer(cfg.act, policy)
    trainer.to_device(device)

    # Compile if requested
    if cfg.torch_compile_mode is not None:
        print(f"Compiling model with mode: {cfg.torch_compile_mode}")
        trainer.compile(mode=cfg.torch_compile_mode)

    # Create checkpoint directory (same format as cal-ql)
    if cfg.save_every_n_epoch > 0:
        ckpt_dir = os.path.join(cfg.ckpt_path, f'{cfg.logging.prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {cfg.num_epochs} epochs")
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_metrics = {
            'epoch': epoch,
            'act/total_loss': 0,
            'act/recon_loss': 0,
            'act/kl_loss': 0,
        }
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.num_epochs}")
        for batch in pbar:
            # Train step
            metrics = trainer.train(batch)
            global_step += 1

            # Accumulate metrics
            for k in ['act/total_loss', 'act/recon_loss', 'act/kl_loss']:
                if k in metrics:
                    epoch_metrics[k] += metrics[k].item() if hasattr(metrics[k], 'item') else metrics[k]
            num_batches += 1

            # Update progress bar
            total_loss = metrics['act/total_loss'].item() if hasattr(metrics['act/total_loss'], 'item') else metrics['act/total_loss']
            recon_loss = metrics['act/recon_loss'].item() if hasattr(metrics['act/recon_loss'], 'item') else metrics['act/recon_loss']
            kl_loss = metrics['act/kl_loss'].item() if hasattr(metrics['act/kl_loss'], 'item') else metrics['act/kl_loss']
            pbar.set_postfix({
                'loss': f"{total_loss:.4f}",
                'recon': f"{recon_loss:.4f}",
                'kl': f"{kl_loss:.4f}",
            })

            # Log per-step metrics
            if global_step % cfg.log_every_n_step == 0:
                wandb_logger.log(metrics, step=global_step)

        # Average epoch metrics
        for k in ['act/total_loss', 'act/recon_loss', 'act/kl_loss']:
            epoch_metrics[k] /= num_batches

        # Step scheduler
        trainer.step_scheduler()
        epoch_metrics['lr'] = trainer.optimizer.param_groups[0]['lr']

        # Log epoch metrics
        wandb_logger.log(epoch_metrics, step=global_step)
        print(f"Epoch {epoch}: loss={epoch_metrics['act/total_loss']:.4f}, "
              f"recon={epoch_metrics['act/recon_loss']:.4f}, "
              f"kl={epoch_metrics['act/kl_loss']:.4f}")

        # Evaluation
        if epoch % cfg.eval_every_n_epoch == 0:
            eval_metrics = trainer.evaluate(dataloader, num_batches=10)
            wandb_logger.log(eval_metrics, step=global_step)
            print(f"  Eval: loss={eval_metrics['eval/total_loss']:.4f}")

        # Save checkpoint (same format as cal-ql)
        if cfg.save_every_n_epoch > 0 and epoch % cfg.save_every_n_epoch == 0:
            ckpt_file_path = os.path.join(ckpt_dir, f'checkpoint_{epoch:05d}.pt')
            trainer.save_checkpoint(ckpt_file_path)

    # Save final checkpoint
    if cfg.save_every_n_epoch > 0:
        final_ckpt_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
        trainer.save_checkpoint(final_ckpt_path)
        print(f"\nTraining complete! Final checkpoint saved to {final_ckpt_path}")


if __name__ == "__main__":
    main()
