import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import json
from datetime import datetime
import argparse
import signal
import sys

from agent import TetrisVisualAgent
from screenshots_utils import ScreenshotProcessor, FrameBuffer
from tetris_env import TetrisEnv


class VisualTetrisTrainer:
    """
    Main training class for visual Tetris RL agent
    """
    
    def __init__(self, config_path="training_config.json"):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.setup_training_params()
        self.setup_components()
        self.setup_tracking()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        self.shutdown_requested = False
    
    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "training": {
                "episodes": 10000,
                "max_steps_per_episode": 1000,
                "save_interval": 100,
                "eval_interval": 50,
                "eval_episodes": 5,
                "render_training": False,
                "render_evaluation": True
            },
            "agent": {
                "input_shape": [4, 84, 84],
                "num_actions": 4,
                "use_amp": True,
                "learning_rate": 0.00025,
                "batch_size": 32,
                "memory_size": 100000,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "target_update_freq": 1000,
                "training_start": 1000
            },
            "preprocessing": {
                "target_size": [84, 84],
                "grayscale": True,
                "frame_stack": 4
            },
            "rewards": {
                "line_clear_multiplier": 1.0,
                "survival_bonus": 0.1,
                "height_penalty": -0.01
            },
            "logging": {
                "log_interval": 10,
                "plot_interval": 100,
                "save_plots": True
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations
                self.merge_config(default_config, user_config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        else:
            print(f"Config file {config_path} not found. Using defaults.")
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        
        return default_config
    
    def merge_config(self, default, user):
        """Recursively merge user config into default config"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self.merge_config(default[key], value)
            else:
                default[key] = value
    
    def setup_directories(self):
        """Create necessary directories for saving models and logs"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"tetris_visual_training_{self.timestamp}"
        
        self.dirs = {
            'models': os.path.join(self.base_dir, 'models'),
            'logs': os.path.join(self.base_dir, 'logs'),
            'plots': os.path.join(self.base_dir, 'plots'),
            'videos': os.path.join(self.base_dir, 'videos')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"Training directory: {self.base_dir}")
    
    def setup_training_params(self):
        """Setup training parameters from config"""
        self.episodes = self.config['training']['episodes']
        self.max_steps = self.config['training']['max_steps_per_episode']
        self.save_interval = self.config['training']['save_interval']
        self.eval_interval = self.config['training']['eval_interval']
        self.eval_episodes = self.config['training']['eval_episodes']
        self.render_training = self.config['training']['render_training']
        self.render_evaluation = self.config['training']['render_evaluation']
        
        self.log_interval = self.config['logging']['log_interval']
        self.plot_interval = self.config['logging']['plot_interval']
    
    def setup_components(self):
        """Initialize agent, environment, and preprocessing components"""
        # Environment
        self.env = TetrisEnv(render_mode=self.render_training)
        self.eval_env = TetrisEnv(render_mode=self.render_evaluation)
        
        # Agent
        agent_config = self.config['agent']
        self.agent = TetrisVisualAgent(
            input_shape=tuple(agent_config['input_shape']),
            num_actions=agent_config['num_actions'],
            use_amp=agent_config['use_amp']
        )
        
        # Override agent parameters with config
        self.agent.batch_size = agent_config['batch_size']
        self.agent.memory = deque(maxlen=agent_config['memory_size'])
        self.agent.epsilon = agent_config['epsilon_start']
        self.agent.epsilon_min = agent_config['epsilon_end']
        self.agent.epsilon_decay = agent_config['epsilon_decay']
        self.agent.target_update_frequency = agent_config['target_update_freq']
        
        # Screenshot processor
        prep_config = self.config['preprocessing']
        self.processor = ScreenshotProcessor(
            target_size=tuple(prep_config['target_size']),
            grayscale=prep_config['grayscale']
        )
        
        # Set environment and processor for the agent
        self.agent.set_env_and_processor(self.env, self.processor)
        
        # Frame buffers
        self.train_frame_buffer = FrameBuffer(capacity=prep_config['frame_stack'])
        self.eval_frame_buffer = FrameBuffer(capacity=prep_config['frame_stack'])
    
    def setup_tracking(self):
        """Setup training progress tracking"""
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_lines_cleared': [],
            'losses': [],
            'q_values': [],
            'epsilons': [],
            'eval_rewards': [],
            'eval_lengths': [],
            'eval_lines_cleared': []
        }
        
        self.best_eval_score = -float('inf')
        self.training_start_time = time.time()
    
    def signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print("\nShutdown requested. Saving progress...")
        self.shutdown_requested = True
    
    def get_state_representation(self, obs, frame_buffer):
        """Convert environment observation to neural network input"""
        # Capture screenshot
        screenshot = self.processor.capture_game_screen(self.env)
        
        # Preprocess
        processed_frame = self.processor.preprocess_screenshot(screenshot)
        
        # Add to frame buffer
        frame_buffer.add_frame(processed_frame)
        
        # Get stacked frames
        stacked_frames = frame_buffer.get_stacked_frames()
        
        return stacked_frames
    
    def calculate_shaped_reward(self, obs, reward, done, lines_cleared):
        """Calculate shaped reward for better learning"""
        shaped_reward = reward
        
        # Line clear bonus
        if lines_cleared > 0:
            shaped_reward += lines_cleared * self.config['rewards']['line_clear_multiplier']
        
        # Survival bonus
        if not done:
            shaped_reward += self.config['rewards']['survival_bonus']
        
        # Height penalty (encourage keeping the board low)
        if obs and 'grid' in obs:
            grid = obs['grid']
            filled_rows = 0
            for row_idx, row in enumerate(grid):
                if np.any(row):
                    filled_rows = len(grid) - row_idx
                    break
            shaped_reward += filled_rows * self.config['rewards']['height_penalty']
        
        return shaped_reward
    
    def train_episode(self, episode):
        """Train for one episode"""
        obs = self.env.reset()
        self.train_frame_buffer.reset()
        
        # Get initial state
        state = self.get_state_representation(obs, self.train_frame_buffer)
        if state is None:
            return 0, 0, 0  # Skip if state is None
        
        total_reward = 0
        steps = 0
        lines_cleared = 0
        episode_losses = []
        
        for step in range(self.max_steps):
            if self.shutdown_requested:
                break

            # Select action
            action = self.agent.select_action(obs, training=True)
            
            # Take step in environment
            next_obs, env_reward, done = self.env.step(action)
            
            # Count lines cleared
            if hasattr(self.env, 'score'):
                current_lines = self.env.score
                if current_lines > lines_cleared:
                    lines_cleared = current_lines
            
            # Get next state
            next_state = self.get_state_representation(next_obs, self.train_frame_buffer)
            
            # Calculate shaped reward
            shaped_reward = self.calculate_shaped_reward(next_obs, env_reward, done, 
                                                       lines_cleared if env_reward > 0 else 0)
            
            # Store experience
            if next_state is not None:
                self.agent.remember(state, action, shaped_reward, next_state, done)
            
            total_reward += env_reward
            steps += 1
            
            # Train agent
            if (self.agent.steps >= self.config['agent']['training_start'] and 
                self.agent.steps % self.agent.update_frequency == 0):
                loss = self.agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            # Update target network
            if self.agent.steps % self.agent.target_update_frequency == 0:
                self.agent.update_target_network()
            
            # Update state
            state = next_state
            self.agent.steps += 1
            
            # Render if enabled
            if self.render_training:
                self.env.render()
                time.sleep(0.01)  # Small delay for visibility
            
            if done:
                break
        
        # Update epsilon
        self.agent.update_epsilon()
        
        # Update stats
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        return total_reward, steps, lines_cleared, avg_loss
    
    def evaluate_agent(self, num_episodes=None):
        """Evaluate agent performance"""
        if num_episodes is None:
            num_episodes = self.eval_episodes
        
        eval_rewards = []
        eval_lengths = []
        eval_lines = []
        
        for eval_ep in range(num_episodes):
            obs = self.eval_env.reset()
            self.eval_frame_buffer.reset()
            
            state = self.get_state_representation(obs, self.eval_frame_buffer)
            if state is None:
                continue
            
            total_reward = 0
            steps = 0
            lines_cleared = 0
            
            for step in range(self.max_steps):
                # Select action (no exploration)
                action = self.agent.select_action(obs, training=False)
                
                # Take step
                next_obs, reward, done = self.eval_env.step(action)
                
                # Count lines
                if hasattr(self.eval_env, 'score'):
                    lines_cleared = self.eval_env.score
                
                total_reward += reward
                steps += 1
                
                # Get next state
                state = self.get_state_representation(next_obs, self.eval_frame_buffer)
                
                # Render if enabled
                if self.render_evaluation:
                    self.eval_env.render()
                    time.sleep(0.02)
                
                if done or state is None:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            eval_lines.append(lines_cleared)
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        avg_lines = np.mean(eval_lines)
        
        return avg_reward, avg_length, avg_lines
    
    def save_checkpoint(self, episode, is_best=False):
        """Save model checkpoint"""
        filename = f"tetris_visual_ep{episode}.pth"
        if is_best:
            filename = f"tetris_visual_best.pth"
        
        filepath = os.path.join(self.dirs['models'], filename)
        
        checkpoint = {
            'episode': episode,
            'agent_state': {
                'model_state_dict': self.agent.q_network.state_dict(),
                'target_state_dict': self.agent.target_network.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'epsilon': self.agent.epsilon,
                'steps': self.agent.steps
            },
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        
        # Load agent state
        self.agent.q_network.load_state_dict(checkpoint['agent_state']['model_state_dict'])
        self.agent.target_network.load_state_dict(checkpoint['agent_state']['target_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['agent_state']['optimizer_state_dict'])
        self.agent.epsilon = checkpoint['agent_state']['epsilon']
        self.agent.steps = checkpoint['agent_state']['steps']
        
        # Load training stats
        self.training_stats = checkpoint['training_stats']
        
        return checkpoint['episode']
    
    def plot_training_progress(self, episode):
        """Create and save training progress plots"""
        if not self.config['logging']['save_plots']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Tetris Visual Training Progress - Episode {episode}')
        
        # Episode rewards
        if self.training_stats['episode_rewards']:
            axes[0, 0].plot(self.training_stats['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        if self.training_stats['episode_lengths']:
            axes[0, 1].plot(self.training_stats['episode_lengths'])
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
        
        # Lines cleared
        if self.training_stats['episode_lines_cleared']:
            axes[0, 2].plot(self.training_stats['episode_lines_cleared'])
            axes[0, 2].set_title('Lines Cleared per Episode')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Lines')
        
        # Training loss
        if self.training_stats['losses']:
            axes[1, 0].plot(self.training_stats['losses'])
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Epsilon decay
        if self.training_stats['epsilons']:
            axes[1, 1].plot(self.training_stats['epsilons'])
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
        
        # Evaluation scores
        if self.training_stats['eval_rewards']:
            axes[1, 2].plot(range(0, len(self.training_stats['eval_rewards']) * self.eval_interval, 
                                self.eval_interval), self.training_stats['eval_rewards'])
            axes[1, 2].set_title('Evaluation Rewards')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Avg Reward')
        
        plt.tight_layout()
        plot_path = os.path.join(self.dirs['plots'], f'training_progress_ep{episode}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, resume_from=None):
        """Main training loop"""
        start_episode = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            start_episode = self.load_checkpoint(resume_from)
            print(f"Resumed training from episode {start_episode}")
        
        print(f"Starting visual Tetris training...")
        print(f"Episodes: {self.episodes}")
        print(f"Device: {self.agent.device}")
        print(f"Using AMP: {self.agent.use_amp}")
        
        try:
            for episode in range(start_episode, self.episodes):
                if self.shutdown_requested:
                    break
                
                # Train episode
                episode_reward, episode_length, lines_cleared, avg_loss = self.train_episode(episode)
                
                # Update stats
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['episode_lengths'].append(episode_length)
                self.training_stats['episode_lines_cleared'].append(lines_cleared)
                self.training_stats['epsilons'].append(self.agent.epsilon)
                
                if avg_loss > 0:
                    self.training_stats['losses'].append(avg_loss)
                
                # Logging
                if episode % self.log_interval == 0:
                    elapsed_time = time.time() - self.training_start_time
                    eps_per_hour = (episode - start_episode + 1) / (elapsed_time / 3600)
                    
                    print(f"Episode {episode:5d} | "
                          f"Reward: {episode_reward:6.1f} | "
                          f"Length: {episode_length:4d} | "
                          f"Lines: {lines_cleared:3d} | "
                          f"Loss: {avg_loss:6.3f} | "
                          f"Epsilon: {self.agent.epsilon:.3f} | "
                          f"Speed: {eps_per_hour:.1f} eps/h")
                
                # Evaluation
                if episode % self.eval_interval == 0 and episode > 0:
                    eval_reward, eval_length, eval_lines = self.evaluate_agent()
                    
                    self.training_stats['eval_rewards'].append(eval_reward)
                    self.training_stats['eval_lengths'].append(eval_length)
                    self.training_stats['eval_lines_cleared'].append(eval_lines)
                    
                    print(f"EVAL  {episode:5d} | "
                          f"Reward: {eval_reward:6.1f} | "
                          f"Length: {eval_length:4.1f} | "
                          f"Lines: {eval_lines:4.1f}")
                    
                    # Save best model
                    if eval_reward > self.best_eval_score:
                        self.best_eval_score = eval_reward
                        self.save_checkpoint(episode, is_best=True)
                        print(f"New best model saved! Score: {eval_reward:.1f}")
                
                # Save checkpoint
                if episode % self.save_interval == 0 and episode > 0:
                    self.save_checkpoint(episode)
                
                # Plot progress
                if episode % self.plot_interval == 0 and episode > 0:
                    self.plot_training_progress(episode)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Final save
            self.save_checkpoint(episode, is_best=False)
            self.plot_training_progress(episode)
            
            # Cleanup
            self.env.close()
            self.eval_env.close()
            
            print(f"\nTraining completed!")
            print(f"Total episodes: {episode - start_episode + 1}")
            print(f"Best evaluation score: {self.best_eval_score:.1f}")
            print(f"Results saved in: {self.base_dir}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train Visual Tetris RL Agent')
    parser.add_argument('--config', type=str, default='training_config.json',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to train (overrides config)')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during training')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (requires --resume)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = VisualTetrisTrainer(config_path=args.config)
    
    # Override config with command line arguments
    if args.episodes:
        trainer.episodes = args.episodes
    if args.render:
        trainer.render_training = True
    
    if args.eval_only:
        if not args.resume:
            print("Error: --eval-only requires --resume argument")
            return
        
        # Load model and evaluate
        trainer.load_checkpoint(args.resume)
        eval_reward, eval_length, eval_lines = trainer.evaluate_agent(num_episodes=10)
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {eval_reward:.2f}")
        print(f"Average Length: {eval_length:.1f}")
        print(f"Average Lines Cleared: {eval_lines:.1f}")
    else:
        # Start training
        trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()