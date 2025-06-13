import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from collections import deque, namedtuple
import random
import cv2

# Define a transition tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Add these constants at the top, after imports
GRID_WIDTH = 10
GRID_HEIGHT = 20


class TetrisVisualAgent:
    def __init__(self, input_shape=(4, 84, 84), num_actions=4, use_amp=True):
        """
        Visual Tetris agent that learns from screenshots
        
        Args:
            input_shape: Shape of input (frames, height, width)
            num_actions: Number of possible actions
            use_amp: Whether to use automatic mixed precision
        """
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.use_amp = use_amp
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network setup
        self.q_network = TetrisVisualDQN(input_shape, num_actions).to(self.device)
        self.target_network = TetrisVisualDQN(input_shape, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training parameters
        self.memory = deque(maxlen=100000)  # Larger buffer for visual learning
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001  # For soft target updates
        
        # Counters and tracking
        self.steps = 0
        self.update_frequency = 4  # Update network every 4 steps
        self.target_update_frequency = 1000  # Update target network every 1000 steps
        self.recent_rewards = deque(maxlen=100)
        self.training_stats = {'losses': [], 'q_values': [], 'epsilons': []}
        
        # Optimizer and loss
        self.scaler = GradScaler(enabled=use_amp)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=2.5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=100
        )
        
        # Action mapping
        self.action_map = {
            0: 'left',
            1: 'right', 
            2: 'rotate',
            3: 'drop'
        }

    def set_env_and_processor(self, env, processor):
        """Set environment and screenshot processor references for the agent."""
        self.env = env
        self.processor = processor

    def select_action(self, state, training=True):
        """Select action using placement-based approach"""
        # Get current and next pieces from environment
        current_piece = self.env.current_piece.copy()
        next_piece = self.env.get_next_piece()
        if next_piece is None:
            # Fallback: use current_piece or a random shape
            next_piece = self.env.current_piece.copy()
        
        # Generate possible placements
        placements = self.generate_placements(state['grid'], current_piece)
        
        if not placements:
            return random.randint(0, self.num_actions - 1)  # Fallback
        
        # Evaluate each placement
        best_value = -float('inf')
        best_placement = None
        
        for placement in placements:
            rot, col = placement
            # Simulate placement
            new_grid, lines_cleared = self.env.simulate_placement(
                current_piece, rot, col
            )
            
            # Create next state representation
            next_state = {
                'grid': new_grid,
                'current_piece': next_piece,
                'current_pos': (0, GRID_WIDTH // 2 - len(next_piece[0]) // 2)
            }
            
            # Render state to image
            state_img = self.processor.render_game_state(next_state)
            processed = self.processor.preprocess_screenshot(state_img)
            # Stack 4 identical frames if not enough history (for simulated placements)
            stacked = torch.cat([processed for _ in range(4)], dim=0)  # [4, 84, 84]
            state_tensor = stacked.unsqueeze(0).to(self.device)         # [1, 4, 84, 84]
            
            # Get state value
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        q_values = self.q_network(state_tensor)
                        value = q_values.max(1)[0].item()  # Take the max Q-value for this state
                else:
                    q_values = self.q_network(state_tensor)
                    value = q_values.max(1)[0].item()  # Take the max Q-value for this state
            
            # Add reward for cleared lines
            if lines_cleared > 0:
                value += lines_cleared * 20  # Reward for clearing lines
            
            if value > best_value:
                best_value = value
                best_placement = placement
        
        # Convert placement to action sequence
        actions = self.placement_to_actions(best_placement)
        # Return only the first action for compatibility with DQN
        return actions[0] if isinstance(actions, list) else actions

    def generate_placements(self, grid, piece):
        """Generate all valid placements for a piece"""
        placements = []
        for rotation in range(4):
            rotated_piece = np.rot90(piece, -rotation)
            for col in range(GRID_WIDTH - len(rotated_piece[0]) + 1):
                if not self.env._collision_at_pos(rotated_piece, (0, col), grid):
                    placements.append((rotation, col))
        return placements

    def placement_to_actions(self, placement):
        """Convert placement to sequence of low-level actions"""
        rotation, column = placement
        actions = []
        
        # Rotations
        for _ in range(rotation):
            actions.append(2)  # Rotate action
        
        # Movement
        current_col = GRID_WIDTH // 2 - len(self.env.current_piece[0]) // 2
        move_count = column - current_col
        
        if move_count > 0:
            for _ in range(move_count):
                actions.append(1)  # Move right
        elif move_count < 0:
            for _ in range(abs(move_count)):
                actions.append(0)  # Move left
        
        # Final drop
        actions.append(3)  # Hard drop
        
        return actions

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train the network on a batch of experiences
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.bool)
        
        # Handle next states (some might be None for terminal states)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(self.device)
        
        # Compute Q-values and loss with mixed precision
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.use_amp):
            # Current Q-values
            current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
            
            # Target Q-values
            next_q_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                if len(non_final_next_states) > 0:
                    if self.use_amp:
                        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                            next_vals = self.target_network(non_final_next_states).max(1)[0]
                    else:
                        next_vals = self.target_network(non_final_next_states).max(1)[0]
                    # Ensure dtype matches
                    next_q_values[non_final_mask] = next_vals.float()
            
            # Compute target Q-values
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
            target_q_values = target_q_values.unsqueeze(1)
            
            # Compute loss (Huber loss for stability)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update training stats
        self.training_stats['losses'].append(loss.item())
        avg_q_value = current_q_values.mean().item()
        self.training_stats['q_values'].append(avg_q_value)
        self.training_stats['epsilons'].append(self.epsilon)
        
        # Update learning rate based on loss
        if len(self.training_stats['losses']) % 100 == 0:
            avg_loss = np.mean(self.training_stats['losses'][-100:])
            self.scheduler.step(avg_loss)
        
        return loss.item()
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update_target_network(self):
        """Soft update of target network parameters"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def update_epsilon(self):
        """Decay epsilon for exploration"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_stats(self, episode, total_reward):
        """Update training statistics"""
        self.recent_rewards.append(total_reward)
        return np.mean(self.recent_rewards) if self.recent_rewards else 0
    
    def save(self, filename):
        """Save model weights and training state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_stats': self.training_stats
        }, filename)
        print(f"Visual model saved to {filename}")
    
    def load(self, filename):
        """Load model weights and training state"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.training_stats = checkpoint['training_stats']
        print(f"Visual model loaded from {filename}")


class TetrisVisualDQN(nn.Module):
    """
    Deep Q-Network for visual Tetris learning
    Based on the DQN architecture used for Atari games
    """
    
    def __init__(self, input_shape, num_actions):
        super(TetrisVisualDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """Calculate the output size of convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = self._conv_forward(dummy_input)
            return dummy_output.view(1, -1).size(1)
    
    def _conv_forward(self, x):
        """Forward pass through convolutional layers"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Ensure input is float and properly normalized
        x = x.float()
        
        # Convolutional layers
        x = self._conv_forward(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Transition(state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with priorities"""
        if len(self.buffer) < batch_size:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


def test_visual_agent():
    """Test function for visual agent"""
    # Create a dummy visual agent
    agent = TetrisVisualAgent(input_shape=(4, 84, 84), num_actions=4)
    
    # Create dummy input
    dummy_state = torch.randn(4, 84, 84)
    
    # Test action selection
    action = agent.select_action(dummy_state, training=False)
    print(f"Selected action: {action} ({agent.action_map[action]})")
    
    # Test network forward pass
    with torch.no_grad():
        q_values = agent.q_network(dummy_state.unsqueeze(0))
        print(f"Q-values shape: {q_values.shape}")
        print(f"Q-values: {q_values.squeeze().tolist()}")
    
    print("Visual agent test completed successfully!")


if __name__ == "__main__":
    test_visual_agent()