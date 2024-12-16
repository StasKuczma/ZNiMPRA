def random_search_controller(env, state, num_sequences=100, sequence_length=10):
    best_reward = -np.inf
    best_action = None

    for _ in range(num_sequences):
        total_reward = 0
        actions = np.random.uniform(-2, 2, sequence_length)
        temp_state = state.copy()

        for action in actions:
            temp_state, reward, terminated, truncated, _ = env.step([action])
            total_reward += reward
            if terminated or truncated:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            best_action = actions[0]

    return best_action
