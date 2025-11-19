import hydra
import torch

from model.model import ResNetPolicy, ResNetQFunction


@hydra.main(config_path="../config", config_name="train_offline", version_base=None)
def main(cfg):
    policy = ResNetPolicy(
        cfg.observation_dim,
        cfg.action_dim,
        cfg.policy_obs_proj_arch,
        cfg.policy_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        cfg.policy_log_std_multiplier,
        cfg.policy_log_std_offset,
        train_backbone=False,
    )
    q = ResNetQFunction(
        cfg.observation_dim,
        cfg.action_dim,
        cfg.q_obs_proj_arch,
        cfg.q_out_proj_arch,
        cfg.hidden_dim,
        cfg.orthogonal_init,
        train_backbone=False
    )
    
    compiled_policy = torch.compile(policy)
    compiled_q = torch.compile(q)
    
    dummy_obs = torch.randn(1, cfg.observation_dim)
    dummy_action = torch.randn(1, cfg.action_dim)
    image = torch.randn(1, 3, 224, 224)  # Example image tensor

    # forward
    compiled_policy(dummy_obs, image)
    print("Policy compilation successful.")
    compiled_q(dummy_obs, image, dummy_action)
    print("Q-function compilation successful.")
    
if __name__ == "__main__":
    main()