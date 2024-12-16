import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Pendulum-v1",render_mode="rgb_array",g=9.81)

state, info = env.reset(seed=42)

env.unwrapped.state = np.array([3.14, 0.5])
obs = env.unwrapped._get_obs()


rollout_lengths = [5, 10, 20, 50] 
num_rollouts = [1, 5, 10, 20]

horizon = 20 
num_simulations = 50 

def random_action():
    return np.random.uniform(-2.0, 2.0)


def simulate_rollout(env, initial_state, rollout_length, num_rollouts):
    total_reward = 0
    for _ in range(num_rollouts):
        env.unwrapped.state = initial_state
        for _ in range(rollout_length):
            action = random_action()
            _, reward, terminated, truncated, _ = env.step([action])
            total_reward += reward
            if terminated or truncated:
                break
    return total_reward

results = np.zeros((len(num_rollouts), len(rollout_lengths)))

for i, n in enumerate(num_rollouts):
    for j, l in enumerate(rollout_lengths):
        state, _ = env.reset(seed=42)
        initial_state = env.unwrapped.state
        cumulative_reward = simulate_rollout(env, initial_state, l, n)
        results[i, j] = cumulative_reward

env.close()

plt.figure(figsize=(10, 6))
plt.imshow(results, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Cumulative Reward")
plt.xticks(range(len(rollout_lengths)), rollout_lengths)
plt.yticks(range(len(num_rollouts)), num_rollouts)
plt.xlabel("Rollout Length")
plt.ylabel("Number of Rollouts")
plt.title("Cumulative Reward Heatmap")
plt.show()