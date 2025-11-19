import time

import torch

from model.model import ResNetPolicy

device = "cuda:0"
policy = ResNetPolicy(observation_dim=6, action_dim=6, obs_proj_arch="256-256", out_proj_arch="256-256", hidden_dim=256, orthogonal_init=True, train_backbone=True)
compiled_policy = torch.compile(policy)
policy.to(device)

image = torch.rand((1, 3, 224, 224), device=device)
print("image shape:", image.shape)
observations = torch.rand((1, 6), device=device)
print("Observations shape: ", observations)

print("Forwarding")
start_time = time.time()
output, log_prob = policy(observations, image, deterministic=False)
print("Output shape:", output.shape)
print("Log prob shape:", log_prob.shape)
print("Time taken: %.4f seconds" % (time.time() - start_time))
output, log_prob = compiled_policy(observations, image, deterministic=False)
start_time = time.time()
output, log_prob = compiled_policy(observations, image, deterministic=False)
print("Time taken (compiled): %.4f seconds" % (time.time() - start_time))

print("---------")
print("Forwarding repeat")
output, log_prob = policy(observations, image, deterministic=False, repeat=5)
print("Output shape:", output.shape)
print("Log prob shape:", log_prob.shape)





