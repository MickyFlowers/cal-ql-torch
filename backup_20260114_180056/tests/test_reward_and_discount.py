length = 100
discount = 0.99
step_reward = 0.0
final_reward = 1.0
fail_final_reward = step_reward

final_returns = 0
failed_final_returns = 0

for t in reversed(range(length)):
    if t == length - 1:
        reward = final_reward
    else:
        reward = step_reward
    final_returns = reward + discount * final_returns
print(f"Final returns: {final_returns}")
for t in range(length):
    if t == length - 1:
        reward = fail_final_reward
    else:
        reward = step_reward
    failed_final_returns = reward + discount * failed_final_returns
print(f"Failed Final returns: {failed_final_returns}")
print(f"Difference: {final_returns - failed_final_returns}")
