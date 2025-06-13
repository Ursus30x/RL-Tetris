import numpy as np
from collections import deque
from tetris_env import TetrisEnv,GRID_HEIGHT,GRID_WIDTH,BLOCK_SIZE
import matplotlib.pyplot as plt
import seaborn as sns

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
