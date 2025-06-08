import numpy as np
import torch
from collections import deque
from tetris_env import TetrisEnv  # Your env module
from agent import ImprovedTetrisAgent  # Your agent module

# Normalize grid (e.g., convert 0-7 -> 0.0-1.0)
def normalize_state(obs):
    """Normalize the grid and include the current falling piece"""
    grid = obs["grid"].copy()
    piece = obs["current_piece"]
    pos = obs["current_pos"]

    px, py = pos
    for y in range(piece.shape[0]):
        for x in range(piece.shape[1]):
            if piece[y][x]:
                gx, gy = px + y, py + x
                if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1]:
                    grid[gx][gy] = piece[y][x]  # Use color/ID of the piece

    return grid.astype(np.float32) / 7.0


# Main training loop
def train_tetris_agent(episodes=1000, render_every=200):
    env = TetrisEnv(render_mode=True)
    state_shape = (4, 20, 10)  # 4 stacked frames, each 20x10
    n_actions = 5

    agent = ImprovedTetrisAgent(state_shape=state_shape, n_actions=n_actions)

    for episode in range(episodes):
        raw_state = env.reset()
        agent.reset_frame_stack()
        state = normalize_state(raw_state)
        stacked_state = agent.preprocess_state(state)

        total_reward = 0
        done = False

        while not done:
            action = agent.act(stacked_state, training=True)
            raw_next_state, reward, done = env.step(action)
            next_state = normalize_state(raw_next_state)
            stacked_next_state = agent.preprocess_state(next_state)

            agent.remember(stacked_state, action, reward, stacked_next_state, done)
            loss = agent.replay()

            stacked_state = stacked_next_state
            total_reward += reward

            if episode % render_every == 0:
                env.render()

        # Logging
        print(f"Episode {episode + 1}/{episodes} | Reward: {total_reward} | Epsilon: {agent.epsilon:.4f}")

        agent.training_stats['episodes'].append(episode)
        agent.training_stats['rewards'].append(total_reward)
        agent.training_stats['epsilon'].append(agent.epsilon)
        if loss is not None:
            agent.training_stats['losses'].append(loss)
            agent.training_stats['learning_rates'].append(agent.scheduler.get_last_lr()[0])

    env.close()

if __name__ == "__main__":
    train_tetris_agent(episodes=1000)
