import numpy as np
import torch
from collections import deque
from tetris_env import TetrisEnv  # Your env module
from agent import ImprovedTetrisAgent  # Your agent module
import time
import matplotlib.pyplot as plt
import os
from datetime import datetime

def normalize_state(obs):
    grid = obs["grid"].copy()
    piece = obs["current_piece"]
    pos = obs["current_pos"]
    
    # Oznacz aktualny klocek specjalną wartością (2.0)
    px, py = pos
    for y in range(piece.shape[0]):
        for x in range(piece.shape[1]):
            if piece[y][x]:
                gx, gy = px + y, py + x
                if 0 <= gx < grid.shape[0] and 0 <= gy < grid.shape[1]:
                    grid[gx][gy] = 2.0  # Specjalna wartość dla aktywnego klocka
                    
    # Normalizuj i dodaj kanał wysokości
    normalized = grid.astype(np.float32) / 7.0
    
    # Dodaj kanał z wysokością kolumn
    height_map = np.zeros_like(grid, dtype=np.float32)
    for col in range(grid.shape[1]):
        for row in range(grid.shape[0]):
            if grid[row][col] > 0:
                height = grid.shape[0] - row
                height_map[:, col] = height / grid.shape[0]
                break
    
    # Połącz obie mapy w wielokanałową reprezentację
    return np.stack([normalized, height_map], axis=0)

class TetrisTrainingAnalytics:
    def __init__(self):
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
        fig.suptitle('Tetris DQN Training Analytics', fontsize=16, fontweight='bold')
        
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

def plot_test_results(test_results, model_path):
    """Create visualization of test results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Test Results: {os.path.basename(model_path)}', fontsize=14, fontweight='bold')
    
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

# Main training loop
def train_tetris_agent(episodes=2000, render_every=500, save_every=200, eval_every=100):
    env = TetrisEnv(render_mode=False)  # Wyłącz rendering dla szybszego treningu
    n_frames = 4
    state_shape = (n_frames * 2, 20, 10)  # 2 kanały * 4 klatki
    n_actions = 5

    agent = ImprovedTetrisAgent(state_shape=state_shape, n_actions=n_actions, n_frames=n_frames)
    analytics = TetrisTrainingAnalytics()  # Initialize analytics
    
    # Tracking najlepszych wyników
    best_reward = float('-inf')
    plateau_counter = 0
    
    print("Rozpoczynanie treningu z ulepszonymi nagrodami...")
    print(f"Epsilon start: {agent.epsilon}, min: {agent.epsilon_min}, decay: {agent.epsilon_decay}")

    for episode in range(episodes):
        episode_start_time = time.time()
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
            
            # Trenuj częściej przy dużym buforze
            if len(agent.memory) > agent.warmup_steps:
                loss = agent.replay(episode)
                if loss is not None:
                    agent.training_stats['losses'].append(loss)

            stacked_state = stacked_next_state
            total_reward += reward
            steps_in_episode += 1
            
            # Zlicz usunięte linie
            if hasattr(env, 'score'):
                lines_cleared_total = env.score

            # Renderuj co 50 epizodów
            if episode % render_every == 0:
                if not env.render_mode:
                    env.render_mode = True
                    env.__init__(render_mode=True)
                env.render()
                time.sleep(0.02)  # Mniejsze opóźnienie dla płynności

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        
        # Get current learning rate
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        # Get average value estimate (optional)
        avg_value = None
        if len(agent.training_stats.get('value_estimates', [])) > 0:
            avg_value = np.mean(agent.training_stats['value_estimates'][-100:])
        
        # Update analytics
        analytics.update(
            episode=episode,
            reward=total_reward,
            lines_cleared=lines_cleared_total,
            moves=steps_in_episode,
            duration=episode_duration,
            loss=agent.training_stats['losses'][-1] if agent.training_stats['losses'] else None,
            epsilon=agent.epsilon,
            lr=current_lr,
            avg_reward=agent.update_stats(episode, total_reward),
            memory_size=len(agent.memory),
            avg_value=avg_value
        )
        
        # Aktualizuj statystyki
        avg_reward = agent.update_stats(episode, total_reward)
        
        # Logging co 50 epizodów
        if episode % 50 == 0 or episode < 50:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Avg (50): {avg_reward:.2f}" if len(agent.recent_rewards) >= 50 else "")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Lines: {lines_cleared_total}")
            print(f"  Steps: {steps_in_episode}")
            print(f"  Memory: {len(agent.memory)}")
            if agent.training_stats['losses']:
                print(f"  Loss: {agent.training_stats['losses'][-1]:.4f}")
            print("-" * 40)
        
        # Zapisuj model i analitykę co jakiś czas
        if episode % save_every == 0 and episode > 0:
            filename = f"tetris_model_ep_{episode}.pth"
            agent.save(filename)
            print(f"Model saved: {filename}")
            
            # Save analytics plot
            analytics.plot_training_analytics(f"training_analytics_ep_{episode}.png")
        
        # Sprawdź czy model się poprawia
        if len(agent.recent_rewards) >= 100:
            current_avg = np.mean(list(agent.recent_rewards)[-100:])
            if current_avg > best_reward:
                best_reward = current_avg
                plateau_counter = 0
                agent.save("best_tetris_model.pth")
            else:
                plateau_counter += 1
                
            # Jeśli długo brak postępu, zwiększ eksplorację
            if plateau_counter > 200:
                agent.epsilon = min(0.3, agent.epsilon * 1.2)
                plateau_counter = 0
                print("Plateau detected, increasing exploration")
        
        # Wyłącz rendering po demonstracji
        if episode % render_every == 0 and episode > 0:
            env.render_mode = False

    # Zapisz finalny model
    agent.save("final_tetris_model.pth")
    env.close()
    
    # Save final analytics
    analytics.plot_training_analytics("final_training_analytics.png")
    
    # Pokaż statystyki końcowe
    print("\n" + "="*50)
    print("TRENING ZAKOŃCZONY")
    print(f"Najlepszy średni wynik: {best_reward:.2f}")
    print(f"Finalny epsilon: {agent.epsilon:.4f}")
    print(f"Całkowite kroki: {agent.steps}")
    if agent.training_stats['losses']:
        print(f"Średnia strata: {np.mean(agent.training_stats['losses'][-100:]):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_tetris_agent(episodes=3000)
