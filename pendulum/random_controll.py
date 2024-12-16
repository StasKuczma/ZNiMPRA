import gymnasium as gym
import numpy as np

env = gym.make("Pendulum-v1",render_mode="human",g=9.81)

state, info = env.reset(seed=42)

env.unwrapped.state = np.array([3.14, 0.5])
obs = env.unwrapped._get_obs()


horizon = 20 
num_simulations = 50 

def random_action():
    return np.random.uniform(-2.0, 2.0)


def simulate_ahead(env, state, action_sequence):
    total_reward = 0.0
    env_copy = gym.make("Pendulum-v1", g=9.81) 
    env_copy.reset()
    env_copy.unwrapped.state = state 
    
    for action in action_sequence:

        _, reward, terminated, truncated, _ = env_copy.step([action])
        total_reward += reward
        if terminated or truncated:
            break

    env_copy.close()
    return total_reward


def choose_best_action(env, state, horizon, num_simulations):
    best_action = None
    best_reward = -np.inf

    for _ in range(num_simulations):
        action_sequence = [random_action() for _ in range(horizon)]
        cumulative_reward = simulate_ahead(env, state, action_sequence)

        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            best_action = action_sequence[0]  

    return best_action

for _ in range(200):
    env.render()

    best_action = choose_best_action(env, env.unwrapped.state, horizon, num_simulations)

    state, reward, terminated, truncated, info = env.step([best_action])

    if terminated or truncated:
        break

env.close()