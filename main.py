import numpy as np
import torch
from collections import deque
from tetris_env import TetrisEnv
from agent import TetrisGroupedAgent  # Updated agent module
import time
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

def normalize_state(obs):
    """Create state representation with board and height map"""
    grid = obs["grid"].copy().astype(np.float32)
    
    # Normalize grid values
    normalized_grid = np.where(grid > 0, 1.0, 0.0)
    
    # Create height map
    height_map = np.zeros_like(grid, dtype=np.float32)
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            if grid[row, col] > 0:
                height = grid.shape[0] - row
                height_map[:, col] = height / grid.shape[0]
                break
    
    # Combine both maps into multi-channel representation
    return np.stack([normalized_grid, height_map], axis=0)

def execute_placement_action(env, placement_action):
    """Execute a placement action (rotation, column) in the environment"""
    if placement_action is None:
        return None, -100.0, True  # Game over
    
    rotation, target_col = placement_action
    
    # Get current piece and position
    current_piece = env.current_piece.copy()
    current_pos = env.current_pos.copy()
    
    # Apply rotations
    for _ in range(rotation):
        rotated = env.rotate(current_piece)
        if not env.collision(rotated, current_pos):
            current_piece = rotated
            env.current_piece = rotated
    
    # Move to target column
    target_x = target_col
    current_x = current_pos[1]
    
    # Move piece to target column
    if target_x < current_x:
        # Move left
        for _ in range(current_x - target_x):
            if not env.collision(env.current_piece, (current_pos[0], current_pos[1] - 1)):
                current_pos[1] -= 1
                env.current_pos[1] -= 1
    elif target_x > current_x:
        # Move right
        for _ in range(target_x - current_x):
            if not env.collision(env.current_piece, (current_pos[0], current_pos[1] + 1)):
                current_pos[1] += 1
                env.current_pos[1] += 1
    
    # Hard drop
    lines_cleared = 0
    while not env.collision(env.current_piece, (env.current_pos[0] + 1, env.current_pos[1])):
        env.current_pos[0] += 1
    
    # Place piece and clear lines
    lines_cleared = env.freeze()
    
    # Calculate reward based on lines cleared
    reward = 0.0
    if lines_cleared == 1:
        reward = 40.0
    elif lines_cleared == 2:
        reward = 100.0
    elif lines_cleared == 3:
        reward = 300.0
    elif lines_cleared == 4:
        reward = 1200.0
    
    return env.get_observation(), reward, env.game_over

def train_tetris_agent(episodes=3000, render_every=200, save_every=200):
    # Setup environment
    env = TetrisEnv(render_mode=False)
    state_shape = (2, 20, 10)  # 2 channels: board + height map
    
    # Initialize agent
    agent = TetrisGroupedAgent(
        state_shape=state_shape,
        use_amp=True
    )
    
    # Training metrics
    best_reward = float('-inf')
    plateau_counter = 0
    start_time = time.time()
    
    print("\n" + "="*50)
    print("Starting Grouped Actions DQN Training")
    print(f"Device: {agent.device}")
    print(f"AMP Enabled: {agent.use_amp}")
    print(f"State shape: {state_shape}")
    print(f"Epsilon start: {agent.epsilon}, min: {agent.epsilon_min}")
    print("="*50 + "\n")

    for episode in range(episodes):
        raw_state = env.reset()
        state = normalize_state(raw_state)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
        
        total_reward = 0
        done = False
        moves_made = 0
        total_lines_cleared = 0

        while not done:
            # Get current piece
            current_piece = env.current_piece.copy()
            
            # Select placement action
            placement_action = agent.select_action(raw_state, current_piece, training=True)
            
            # Execute placement
            raw_next_state, reward, done = execute_placement_action(env, placement_action)
            
            if raw_next_state is None:
                break
            
            next_state = normalize_state(raw_next_state)
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(agent.device)
            
            # Store experience
            agent.remember(state_tensor, placement_action, reward, next_state_tensor, done)
            
            # Train
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay(episode)
            
            # Update for next iteration
            state_tensor = next_state_tensor
            raw_state = raw_next_state
            total_reward += reward
            moves_made += 1
            
            # Count lines cleared
            if reward > 0:
                if reward >= 1200:
                    total_lines_cleared += 4
                elif reward >= 300:
                    total_lines_cleared += 3
                elif reward >= 100:
                    total_lines_cleared += 2
                elif reward >= 40:
                    total_lines_cleared += 1
            
            # Render occasionally
            if episode % render_every == 0 and episode > 0:
                if not env.render_mode:
                    env.render_mode = True
                    env.__init__(render_mode=True)
                env.render()
                time.sleep(0.1)

        # Update statistics
        avg_reward = agent.update_stats(episode, total_reward)
        
        # Logging
        if episode % 25 == 0 or episode < 25:
            time_elapsed = (time.time() - start_time) / 60
            print(f"\nEpisode {episode + 1}/{episodes} ({time_elapsed:.1f} min)")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Lines Cleared: {total_lines_cleared}")
            print(f"  Moves Made: {moves_made}")
            if len(agent.recent_rewards) >= 25:
                print(f"  Avg Reward (25): {avg_reward:.1f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Memory Size: {len(agent.memory)}")
            if agent.training_stats['losses']:
                avg_loss = np.mean(agent.training_stats['losses'][-25:])
                print(f"  Avg Loss (25): {avg_loss:.6f}")
            print("-" * 40)
        
        # Save model periodically
        if episode % save_every == 0 and episode > 0:
            filename = f"tetris_grouped_model_ep_{episode}.pth"
            agent.save(filename)
        
        # Check for improvement
        if len(agent.recent_rewards) >= 50:
            current_avg = np.mean(list(agent.recent_rewards)[-50:])
            if current_avg > best_reward:
                best_reward = current_avg
                plateau_counter = 0
                agent.save("best_grouped_tetris_model.pth")
                print(f"✓ New best average reward: {best_reward:.1f}")
            else:
                plateau_counter += 1
                
            # Adjust exploration if plateau detected
            if plateau_counter > 100:
                agent.epsilon = min(0.2, agent.epsilon * 1.5)
                plateau_counter = 0
                print("⚠ Plateau detected, increasing exploration")
        
        # Turn off rendering after demo
        if episode % render_every == 0 and episode > 0:
            env.render_mode = False

    # Save final model
    agent.save("final_grouped_tetris_model.pth")
    env.close()
    
    # Final stats
    total_time = (time.time() - start_time) / 3600
    print("\n" + "="*50)
    print("GROUPED ACTIONS TRAINING COMPLETE")
    print(f"Best 50-episode avg: {best_reward:.1f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total training steps: {agent.steps}")
    print(f"Training time: {total_time:.2f} hours")
    if agent.training_stats['losses']:
        print(f"Final avg loss: {np.mean(agent.training_stats['losses'][-50:]):.6f}")
    print("="*50)

if __name__ == "__main__":
    train_tetris_agent(episodes=3000)