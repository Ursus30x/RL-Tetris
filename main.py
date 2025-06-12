import numpy as np
import torch
from collections import deque
from tetris_env import TetrisEnv
from agent import TetrisGroupedAgent
from heuristic import TetrisHeuristicAgent
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TetrisTrainingAnalytics:
    def __init__(self, model_name):
        self.model_name = model_name
        self.episode_rewards = []
        self.episode_lines_cleared = []
        self.episode_moves = []
        self.episode_durations = []
        self.losses = []
        self.epsilon_history = []
        self.learning_rates = []
        self.avg_rewards_window = []
        self.best_rewards = []
        self.memory_sizes = []
        self.value_estimates = []
        
    def update(self, episode, reward, lines_cleared, moves, duration, loss, epsilon, lr, avg_reward, memory_size, avg_value):
        self.episode_rewards.append(reward)
        self.episode_lines_cleared.append(lines_cleared)
        self.episode_moves.append(moves)
        self.episode_durations.append(duration)
        if loss is not None:
            self.losses.append(loss)
        self.epsilon_history.append(epsilon)
        self.learning_rates.append(lr)
        self.avg_rewards_window.append(avg_reward)
        self.memory_sizes.append(memory_size)
        if avg_value is not None:
            self.value_estimates.append(avg_value)
    
    def plot_training_analytics(self, save_path="training_analytics.png"):
        """Create comprehensive training analytics plots"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Tetris {self.model_name} Training Analytics', fontsize=16, fontweight='bold')
        
        # 1. Episode Rewards
        axes[0,0].plot(self.episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
        if len(self.avg_rewards_window) > 0:
            axes[0,0].plot(self.avg_rewards_window, color='red', linewidth=2, label='Moving Average (100)')
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 2. Lines Cleared
        axes[0,1].plot(self.episode_lines_cleared, color='green', alpha=0.7)
        if len(self.episode_lines_cleared) > 100:
            # Moving average for lines cleared
            window = 50
            lines_ma = [np.mean(self.episode_lines_cleared[max(0, i-window):i+1]) 
                       for i in range(window, len(self.episode_lines_cleared))]
            axes[0,1].plot(range(window, len(self.episode_lines_cleared)), lines_ma, 
                          color='darkgreen', linewidth=2, label=f'MA({window})')
        axes[0,1].set_title('Lines Cleared per Episode')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Lines Cleared')
        axes[0,1].grid(True, alpha=0.3)
        if len(self.episode_lines_cleared) > 100:
            axes[0,1].legend()
        
        # 3. Episode Duration (Moves)
        axes[0,2].plot(self.episode_moves, color='orange', alpha=0.7)
        if len(self.episode_moves) > 100:
            moves_ma = [np.mean(self.episode_moves[max(0, i-50):i+1]) 
                       for i in range(50, len(self.episode_moves))]
            axes[0,2].plot(range(50, len(self.episode_moves)), moves_ma, 
                          color='darkorange', linewidth=2, label='MA(50)')
        axes[0,2].set_title('Moves per Episode')
        axes[0,2].set_xlabel('Episode')
        axes[0,2].set_ylabel('Number of Moves')
        axes[0,2].grid(True, alpha=0.3)
        if len(self.episode_moves) > 100:
            axes[0,2].legend()
        
        # 4. Training Loss
        if self.losses:
            axes[1,0].plot(self.losses, color='red', alpha=0.6)
            if len(self.losses) > 100:
                loss_ma = [np.mean(self.losses[max(0, i-100):i+1]) 
                          for i in range(100, len(self.losses))]
                axes[1,0].plot(range(100, len(self.losses)), loss_ma, 
                              color='darkred', linewidth=2, label='MA(100)')
            axes[1,0].set_title('Training Loss')
            axes[1,0].set_xlabel('Training Step')
            axes[1,0].set_ylabel('Loss')
            axes[1,0].set_yscale('log')
            axes[1,0].grid(True, alpha=0.3)
            if len(self.losses) > 100:
                axes[1,0].legend()
        
        # 5. Epsilon Decay
        axes[1,1].plot(self.epsilon_history, color='purple', linewidth=2)
        axes[1,1].set_title('Epsilon Decay (Exploration Rate)')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Epsilon')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Learning Rate
        if self.learning_rates:
            axes[1,2].plot(self.learning_rates, color='brown', linewidth=2)
            axes[1,2].set_title('Learning Rate Schedule')
            axes[1,2].set_xlabel('Episode')
            axes[1,2].set_ylabel('Learning Rate')
            axes[1,2].set_yscale('log')
            axes[1,2].grid(True, alpha=0.3)
        
        # 7. Value Estimates
        if self.value_estimates:
            axes[2,0].plot(self.value_estimates, color='cyan', alpha=0.7)
            if len(self.value_estimates) > 100:
                value_ma = [np.mean(self.value_estimates[max(0, i-100):i+1]) 
                           for i in range(100, len(self.value_estimates))]
                axes[2,0].plot(range(100, len(self.value_estimates)), value_ma, 
                              color='darkcyan', linewidth=2, label='MA(100)')
            axes[2,0].set_title('Average Value Estimates')
            axes[2,0].set_xlabel('Training Step')
            axes[2,0].set_ylabel('Value')
            axes[2,0].grid(True, alpha=0.3)
            if len(self.value_estimates) > 100:
                axes[2,0].legend()
        
        # 8. Memory Usage
        axes[2,1].plot(self.memory_sizes, color='magenta', linewidth=2)
        axes[2,1].set_title('Replay Buffer Size')
        axes[2,1].set_xlabel('Episode')
        axes[2,1].set_ylabel('Memory Size')
        axes[2,1].grid(True, alpha=0.3)
        
        # 9. Performance Distribution
        if len(self.episode_rewards) > 100:
            recent_rewards = self.episode_rewards[-500:] if len(self.episode_rewards) > 500 else self.episode_rewards
            axes[2,2].hist(recent_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[2,2].axvline(np.mean(recent_rewards), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(recent_rewards):.1f}')
            axes[2,2].set_title('Recent Reward Distribution')
            axes[2,2].set_xlabel('Reward')
            axes[2,2].set_ylabel('Frequency')
            axes[2,2].legend()
            axes[2,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training analytics saved to {save_path}")

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

def execute_placement_action(env, placement_action, heuristics=False):
    """Execute a placement action (rotation, column) in the environment"""
    if placement_action is None:
        return None, -50.0, True  # Game over penalty
    
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
    
    # Unified reward calculation
    reward = 0.0
    if lines_cleared == 1:
        reward = 20.0
    elif lines_cleared == 2:
        reward = 60.0
    elif lines_cleared == 3:
        reward = 200.0
    elif lines_cleared == 4:
        reward = 500.0
    
    return env.get_observation(), reward, env.game_over

def test_trained_model(model_path, num_games=10, render=False):
    """Test a trained model and return performance statistics"""
    print(f"\n{'='*50}")
    print("TESTING TRAINED MODEL")
    print(f"Model: {model_path}")
    print(f"Number of test games: {num_games}")
    print(f"{'='*50}\n")
    
    # Setup environment
    env = TetrisEnv(render_mode=render)
    state_shape = (2, 20, 10)
    
    # Initialize agent
    agent = TetrisGroupedAgent(state_shape=state_shape, use_amp=True)
    
    # Load trained model
    try:
        agent.load(model_path)
        agent.epsilon = 0.0  # No exploration during testing
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Test statistics
    test_results = {
        'rewards': [],
        'lines_cleared': [],
        'moves': [],
        'max_reward': float('-inf'),
        'min_reward': float('inf'),
        'total_lines': 0
    }
    
    for game in range(num_games):
        raw_state = env.reset()
        state = normalize_state(raw_state)
        
        total_reward = 0
        total_lines_cleared = 0
        moves_made = 0
        done = False
        
        print(f"\nGame {game + 1}/{num_games}")
        print("-" * 30)
        
        while not done:
            # Get current piece
            current_piece = env.current_piece.copy()
            
            # Select best action (no exploration)
            placement_action = agent.select_action(raw_state, current_piece, training=False)
            
            # Execute placement
            raw_next_state, reward, done = execute_placement_action(env, placement_action)
            
            if raw_next_state is None:
                break
            
            # Update for next iteration
            raw_state = raw_next_state
            total_reward += reward
            moves_made += 1
            
            # Count lines cleared - FIXED VALUES
            if reward > 0:
                if reward >= 500:
                    lines_this_move = 4
                elif reward >= 200:
                    lines_this_move = 3
                elif reward >= 60:
                    lines_this_move = 2
                elif reward >= 20:
                    lines_this_move = 1
                else:
                    lines_this_move = 0
                total_lines_cleared += lines_this_move
                print(f"  Lines cleared: {lines_this_move} (Total: {total_lines_cleared})")
            
            # Render if enabled
            if render:
                env.render()
                time.sleep(0.05)  # Slower for better viewing
        
        # Store results
        test_results['rewards'].append(total_reward)
        test_results['lines_cleared'].append(total_lines_cleared)
        test_results['moves'].append(moves_made)
        test_results['total_lines'] += total_lines_cleared
        test_results['max_reward'] = max(test_results['max_reward'], total_reward)
        test_results['min_reward'] = min(test_results['min_reward'], total_reward)
        
        print(f"  Final Score: {total_reward:.1f}")
        print(f"  Lines Cleared: {total_lines_cleared}")
        print(f"  Moves Made: {moves_made}")
        print(f"  Efficiency: {total_lines_cleared/moves_made:.3f} lines/move" if moves_made > 0 else "  Efficiency: 0.000")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(test_results['rewards'])
    std_reward = np.std(test_results['rewards'])
    avg_lines = np.mean(test_results['lines_cleared'])
    avg_moves = np.mean(test_results['moves'])
    
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Average Reward: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"Best Game: {test_results['max_reward']:.1f}")
    print(f"Worst Game: {test_results['min_reward']:.1f}")
    print(f"Average Lines Cleared: {avg_lines:.1f}")
    print(f"Total Lines Cleared: {test_results['total_lines']}")
    print(f"Average Moves per Game: {avg_moves:.1f}")
    print(f"Overall Efficiency: {test_results['total_lines']/(sum(test_results['moves'])):.3f} lines/move")
    print(f"{'='*50}")
    
    # Create test results visualization
    plot_test_results(test_results, model_path)
    
    return test_results

def plot_test_results(test_results, model_name):
    """Create visualization of test results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Test Results: {os.path.basename(model_name)}', fontsize=14, fontweight='bold')
    
    games = list(range(1, len(test_results['rewards']) + 1))
    
    # 1. Rewards per game
    axes[0,0].bar(games, test_results['rewards'], color='skyblue', alpha=0.7, edgecolor='navy')
    axes[0,0].axhline(np.mean(test_results['rewards']), color='red', linestyle='--', 
                     label=f'Average: {np.mean(test_results["rewards"]):.1f}')
    axes[0,0].set_title('Rewards per Game')
    axes[0,0].set_xlabel('Game')
    axes[0,0].set_ylabel('Total Reward')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Lines cleared per game
    axes[0,1].bar(games, test_results['lines_cleared'], color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    axes[0,1].axhline(np.mean(test_results['lines_cleared']), color='red', linestyle='--',
                     label=f'Average: {np.mean(test_results["lines_cleared"]):.1f}')
    axes[0,1].set_title('Lines Cleared per Game')
    axes[0,1].set_xlabel('Game')
    axes[0,1].set_ylabel('Lines Cleared')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Moves per game
    axes[1,0].bar(games, test_results['moves'], color='orange', alpha=0.7, edgecolor='darkorange')
    axes[1,0].axhline(np.mean(test_results['moves']), color='red', linestyle='--',
                     label=f'Average: {np.mean(test_results["moves"]):.1f}')
    axes[1,0].set_title('Moves per Game')
    axes[1,0].set_xlabel('Game')
    axes[1,0].set_ylabel('Number of Moves')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Efficiency (lines/move)
    efficiency = [lines/moves if moves > 0 else 0 
                 for lines, moves in zip(test_results['lines_cleared'], test_results['moves'])]
    axes[1,1].bar(games, efficiency, color='purple', alpha=0.7, edgecolor='purple')
    axes[1,1].axhline(np.mean(efficiency), color='red', linestyle='--',
                     label=f'Average: {np.mean(efficiency):.3f}')
    axes[1,1].set_title('Efficiency (Lines/Move)')
    axes[1,1].set_xlabel('Game')
    axes[1,1].set_ylabel('Lines per Move')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"test_results_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Test results visualization saved to {save_path}")

def train_tetris_agent(episodes=3000, render_every=200, save_every=200, test_during_training=True):
    # Setup environment
    env = TetrisEnv(render_mode=False)
    state_shape = (2, 20, 10)  # 2 channels: board + height map
    
    # Initialize agent
    agent = TetrisGroupedAgent(
        state_shape=state_shape,
        use_amp=True
    )
    
    # Initialize analytics
    analytics = TetrisTrainingAnalytics("FP16 Value-based Agent")
    
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
        episode_start_time = time.time()
        
        raw_state = env.reset()
        state = normalize_state(raw_state)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
        
        total_reward = 0
        done = False
        moves_made = 0
        total_lines_cleared = 0
        episode_losses = []

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
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update for next iteration
            state_tensor = next_state_tensor
            raw_state = raw_next_state
            total_reward += reward
            moves_made += 1
            
            # Count lines cleared - FIXED VALUES
            if reward > 0:
                if reward >= 500:
                    total_lines_cleared += 4
                elif reward >= 200:
                    total_lines_cleared += 3
                elif reward >= 60:
                    total_lines_cleared += 2
                elif reward >= 20:
                    total_lines_cleared += 1
            
            # Render occasionally
            if episode % render_every == 0 and episode > 0:
                if not env.render_mode:
                    env.render_mode = True
                    env.__init__(render_mode=True)
                env.render()
                time.sleep(0.1)

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        
        # Update statistics
        avg_reward = agent.update_stats(episode, total_reward)
        
        # Get current learning rate
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        # Calculate average loss for this episode
        avg_loss = np.mean(episode_losses) if episode_losses else None
        avg_value = np.mean(agent.training_stats['values'][-10:]) if agent.training_stats['values'] else None
        
        # Update analytics
        analytics.update(
            episode=episode,
            reward=total_reward,
            lines_cleared=total_lines_cleared,
            moves=moves_made,
            duration=episode_duration,
            loss=avg_loss,
            epsilon=agent.epsilon,
            lr=current_lr,
            avg_reward=avg_reward,
            memory_size=len(agent.memory),
            avg_value=avg_value
        )
        
        # Logging
        if episode % 25 == 0 or episode < 25:
            time_elapsed = (time.time() - start_time) / 60
            print(f"\nEpisode {episode + 1}/{episodes} ({time_elapsed:.1f} min)")
            print(f"  Total Reward: {total_reward:.1f}")
            print(f"  Lines Cleared: {total_lines_cleared}")
            print(f"  Moves Made: {moves_made}")
            print(f"  Episode Duration: {episode_duration:.2f}s")
            if len(agent.recent_rewards) >= 25:
                print(f"  Avg Reward (25): {avg_reward:.1f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Memory Size: {len(agent.memory)}")
            if avg_loss is not None:
                print(f"  Avg Loss: {avg_loss:.6f}")
            if avg_value is not None:
                print(f"  Avg Value: {avg_value:.3f}")
            print("-" * 40)
        
        # Save model periodically
        if episode % save_every == 0 and episode > 0:
            filename = f"tetris_grouped_model_ep_{episode}.pth"
            agent.save(filename)
        
        # Test during training
        if test_during_training and episode % (episodes // 5) == 0 and episode > 0:
            print(f"\n🧪 Running test at episode {episode}...")
            test_results = test_trained_model(f"tetris_grouped_model_ep_{episode}.pth", 
                                            num_games=3, render=False)
        
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
        
        # Generate analytics plots periodically
        if episode % 500 == 0 and episode > 0:
            analytics.plot_training_analytics(f"training_progress_ep_{episode}.png")

    # Save final model
    agent.save("final_grouped_tetris_model.pth")
    env.close()
    
    # Final analytics
    analytics.plot_training_analytics("final_training_analytics.png")
    
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
    
    return analytics

def test_heuristic_agent(num_games=10, render=False):
    """Test the heuristic agent's performance using unified scoring"""
    print("\nTesting Heuristic Agent...")
    
    env = TetrisEnv(render_mode=render)
    agent = TetrisHeuristicAgent()
    
    results = {
        'rewards': [],
        'lines_cleared': [],
        'moves': [],
        'max_reward': float('-inf'),
        'min_reward': float('inf'),
        'total_lines': 0
    }
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        total_reward = 0
        total_lines_cleared = 0
        moves_made = 0
        
        print(f"\nGame {game + 1}/{num_games}")
        print("-" * 30)
        
        while not done:
            rotation, col = agent.get_action(env)
            
            # Execute the placement
            for _ in range(rotation):
                env.current_piece = env.rotate(env.current_piece)
            env.current_pos[1] = col
            
            # Hard drop
            while not env.collision(env.current_piece, (env.current_pos[0] + 1, env.current_pos[1])):
                env.current_pos[0] += 1
            
            # Freeze piece and get lines cleared
            lines_cleared = env.freeze()
            moves_made += 1
            
            # Calculate reward based on lines cleared (same as RL agent)
            reward = 0.0
            if lines_cleared == 1:
                reward = 20.0
            elif lines_cleared == 2:
                reward = 60.0
            elif lines_cleared == 3:
                reward = 200.0
            elif lines_cleared == 4:
                reward = 500.0
            
            total_reward += reward
            total_lines_cleared += lines_cleared
            
            if render:
                env.render()
                #time.sleep(0.05)
            
            done = env.game_over
            
            if lines_cleared > 0:
                print(f"  Lines cleared: {lines_cleared} (Total: {total_lines_cleared})")
        
        # Store results
        results['rewards'].append(total_reward)
        results['lines_cleared'].append(total_lines_cleared)
        results['moves'].append(moves_made)
        results['total_lines'] += total_lines_cleared
        results['max_reward'] = max(results['max_reward'], total_reward)
        results['min_reward'] = min(results['min_reward'], total_reward)
        
        print(f"  Final Score: {total_reward:.1f}")
        print(f"  Lines Cleared: {total_lines_cleared}")
        print(f"  Moves Made: {moves_made}")
        print(f"  Efficiency: {total_lines_cleared/moves_made:.3f} lines/move" if moves_made > 0 else "  Efficiency: 0.000")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(results['rewards'])
    std_reward = np.std(results['rewards'])
    avg_lines = np.mean(results['lines_cleared'])
    avg_moves = np.mean(results['moves'])
    
    print(f"\n{'='*50}")
    print("HEURISTIC AGENT RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Average Reward: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"Best Game: {results['max_reward']:.1f}")
    print(f"Worst Game: {results['min_reward']:.1f}")
    print(f"Average Lines Cleared: {avg_lines:.1f}")
    print(f"Total Lines Cleared: {results['total_lines']}")
    print(f"Average Moves per Game: {avg_moves:.1f}")
    print(f"Overall Efficiency: {results['total_lines']/(sum(results['moves'])):.3f} lines/move")
    print(f"{'='*50}")
    
    # Create test results visualization
    plot_test_results(results, "heuristic_agent")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tetris Training and Testing')
    parser.add_argument('--mode', choices=['train', 'test', 'both', 'heuristic'], default='both',
                       help='Mode: train, test, or both')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes')
    parser.add_argument('--model_path', type=str, default='best_grouped_tetris_model.pth',
                       help='Path to model for testing')
    parser.add_argument('--test_games', type=int, default=10,
                       help='Number of games to test')
    parser.add_argument('--render', action='store_true',
                       help='Render during testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        analytics = train_tetris_agent(episodes=args.episodes)
    elif args.mode == 'test':
        test_trained_model(args.model_path, num_games=args.test_games, render=args.render)
    elif args.mode == 'heuristic':
        test_heuristic_agent(render=args.render)
    elif args.mode == 'both':  # both
        print("🚀 Starting training phase...")
        analytics = train_tetris_agent(episodes=args.episodes)
        
        print("\n🎮 Starting testing phase...")
        test_trained_model("best_grouped_tetris_model.pth", 
                          num_games=args.test_games, render=args.render)
    else:
        print("no args")