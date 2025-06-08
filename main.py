import numpy as np
import torch
from collections import deque
from tetris_env import TetrisEnv
from agent import ImprovedTetrisAgent  # Updated agent module
import time
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

def normalize_state(obs):
    grid = obs["grid"].copy()
    piece = obs["current_piece"]
    pos = obs["current_pos"]
    
    # Mark current piece with special value (2.0)
    px, py = pos
    for y in range(piece.shape[0]):
        for x in range(piece.shape[1]):
            if piece[y][x]:
                gx, gy = px + y, py + x
                if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1]:
                    grid[gx][gy] = 2.0  # Special value for active piece
                    
    # Normalize and add height channel
    normalized = grid.astype(np.float32) / 7.0
    
    # Add column height channel
    height_map = np.zeros_like(grid, dtype=np.float32)
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            if grid[row][col] > 0:
                height = grid.shape[0] - row
                height_map[:, col] = height / grid.shape[0]
                break
    
    # Combine both maps into multi-channel representation
    return np.stack([normalized, height_map], axis=0)

def train_tetris_agent(episodes=5000, render_every=500, save_every=100, eval_every=100):
        # Setup environment
    env = TetrisEnv(render_mode=False)
    n_frames = 4
    single_frame_shape = (2, 20, 10)  # Single frame shape (2 channels)
    n_actions = 5

    # Initialize agent with single frame shape
    agent = ImprovedTetrisAgent(
        state_shape=single_frame_shape,  # Pass single frame shape here
        n_actions=n_actions,
        n_frames=n_frames,
        use_amp=True
    )
    
    # Training metrics
    best_reward = float('-inf')
    plateau_counter = 0
    start_time = time.time()
    
    print("\n" + "="*50)
    print("Starting FP16 DQN Training")
    print(f"Device: {agent.device}")
    print(f"AMP Enabled: {agent.use_amp}")
    print(f"State shape: {single_frame_shape}")
    print(f"Epsilon start: {agent.epsilon}, min: {agent.epsilon_min}")
    print("="*50 + "\n")

    for episode in range(episodes):
        raw_state = env.reset()
        agent.reset_frame_stack()
        state = normalize_state(raw_state)
        stacked_state = agent.preprocess_state(state)

        total_reward = 0
        done = False
        steps_in_episode = 0
        lines_cleared_total = 0

        while not done:
            action = agent.act(stacked_state, training=True)
            raw_next_state, reward, done = env.step(action)
            next_state = normalize_state(raw_next_state)
            stacked_next_state = agent.preprocess_state(next_state)

            agent.remember(stacked_state, action, reward, stacked_next_state, done)
            
            # Train more frequently when buffer is full
            if len(agent.memory) > agent.warmup_steps:
                loss = agent.replay(episode)
                if loss is not None:
                    agent.training_stats['losses'].append(loss)

            stacked_state = stacked_next_state
            total_reward += reward
            steps_in_episode += 1
            
            # Track lines cleared
            if hasattr(env, 'score'):
                lines_cleared_total = env.score

            # Render occasionally
            if episode % render_every == 0:
                if not env.render_mode:
                    env.render_mode = True
                    env.__init__(render_mode=True)
                env.render()
                time.sleep(0.02)

        # Update statistics
        avg_reward = agent.update_stats(episode, total_reward)
        
        # Logging
        if episode % 50 == 0 or episode < 50:
            time_elapsed = (time.time() - start_time) / 60
            print(f"\nEpisode {episode + 1}/{episodes} ({time_elapsed:.1f} min)")
            print(f"  Reward: {total_reward:.2f}")
            if len(agent.recent_rewards) >= 50:
                print(f"  Avg (50): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Lines: {lines_cleared_total}")
            print(f"  Steps: {steps_in_episode}")
            print(f"  Memory: {len(agent.memory)}")
            if agent.training_stats['losses']:
                avg_loss = np.mean(agent.training_stats['losses'][-50:])
                print(f"  Loss (50): {avg_loss:.4f}")
            print("-" * 40)
        
        # Save model periodically
        if episode % save_every == 0 and episode > 0:
            filename = f"tetris_fp16_model_ep_{episode}.pth"
            agent.save(filename)
        
        # Check for improvement
        if len(agent.recent_rewards) >= 100:
            current_avg = np.mean(list(agent.recent_rewards)[-100:])
            if current_avg > best_reward:
                best_reward = current_avg
                plateau_counter = 0
                agent.save("best_fp16_tetris_model.pth")
                print(f"New best avarage: {best_reward}")
            else:
                plateau_counter += 1
                
            # Adjust exploration if plateau detected
            if plateau_counter > 200:
                agent.epsilon = min(0.3, agent.epsilon * 1.2)
                plateau_counter = 0
                print("Plateau detected, increasing exploration")
        
        # Turn off rendering after demo
        if episode % render_every == 0 and episode > 0:
            env.render_mode = False

    # Save final model
    agent.save("final_fp16_tetris_model.pth")
    env.close()
    
    # Final stats
    total_time = (time.time() - start_time) / 3600
    print("\n" + "="*50)
    print("FP16 TRAINING COMPLETE")
    print(f"Best 100-episode avg: {best_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total steps: {agent.steps}")
    print(f"Training time: {total_time:.2f} hours")
    if agent.training_stats['losses']:
        print(f"Final avg loss: {np.mean(agent.training_stats['losses'][-100:]):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_tetris_agent(episodes=5000)