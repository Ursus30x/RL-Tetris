import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from collections import deque, namedtuple
import random

# Define a transition tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class TetrisGroupedAgent:
    def __init__(self, state_shape, use_amp=True):
        # State shape for board only (no frame stacking needed for placement-based)
        self.state_shape = state_shape  # (2, 20, 10) - board + height map
        self.use_amp = use_amp
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network setup - value network (not Q-network, outputs single value)
        self.model = TetrisValueNet(self.state_shape[0]).to(self.device)
        self.target_model = TetrisValueNet(self.state_shape[0]).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Training parameters
        self.memory = deque(maxlen=100000)  # Smaller buffer since each transition is more meaningful
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1  # Much lower epsilon for exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.tau = 0.005
        
        self.steps = 0
        self.recent_rewards = deque(maxlen=100)
        self.training_stats = {'losses': [], 'values': []}
        
        self.scaler = GradScaler(enabled=use_amp)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)

    def generate_possible_placements(self, board, piece):
        """Generate all valid (rotation, column) placements for current piece"""
        placements = []
        
        for rotation in range(4):
            rotated_piece = self.rotate_piece(piece, rotation)
            piece_width = rotated_piece.shape[1]
            
            for col in range(board.shape[1] - piece_width + 1):
                if self.can_place_piece(board, rotated_piece, col):
                    placements.append((rotation, col))
        
        return placements

    def rotate_piece(self, piece, times):
        """Rotate piece clockwise 'times' number of times"""
        rotated = piece.copy()
        for _ in range(times % 4):
            rotated = np.rot90(rotated, -1)
        return rotated

    def can_place_piece(self, board, piece, col):
        """Check if piece can be placed at given column"""
        # Find where piece would land
        drop_row = self.find_drop_row(board, piece, col)
        if drop_row is None:
            return False
        
        # Check if placement is valid
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y, x] != 0:
                    board_row = drop_row + y
                    board_col = col + x
                    if (board_row >= board.shape[0] or 
                        board_col >= board.shape[1] or 
                        board[board_row, board_col] != 0):
                        return False
        return True

    def find_drop_row(self, board, piece, col):
        """Find the row where piece would land if dropped at column"""
        for row in range(board.shape[0] - piece.shape[0] + 1):
            # Check if piece collides at this row
            collision = False
            for y in range(piece.shape[0]):
                for x in range(piece.shape[1]):
                    if piece[y, x] != 0:
                        if board[row + y, col + x] != 0:
                            collision = True
                            break
                if collision:
                    break
            
            if collision:
                return row - 1 if row > 0 else None
        
        return board.shape[0] - piece.shape[0]  # Can place at bottom

    def simulate_placement(self, board, piece, rotation, col):
        """Simulate placing piece and return resulting board state"""
        rotated_piece = self.rotate_piece(piece, rotation)
        drop_row = self.find_drop_row(board, rotated_piece, col)
        
        if drop_row is None:
            return None
        
        # Create new board with piece placed
        new_board = board.copy()
        for y in range(rotated_piece.shape[0]):
            for x in range(rotated_piece.shape[1]):
                if rotated_piece[y, x] != 0:
                    new_board[drop_row + y, col + x] = 1
        
        # Clear completed lines and return lines cleared
        lines_cleared = self.clear_lines_simulation(new_board)
        
        return new_board, lines_cleared

    def clear_lines_simulation(self, board):
        """Clear completed lines and return number of lines cleared"""
        lines_cleared = 0
        new_rows = []
        
        for row in range(board.shape[0]):
            if not np.all(board[row]):  # Line not complete
                new_rows.append(board[row].copy())
            else:
                lines_cleared += 1
        
        # Add empty rows at top
        while len(new_rows) < board.shape[0]:
            new_rows.insert(0, np.zeros(board.shape[1]))
        
        # Update board
        for i, row in enumerate(new_rows):
            board[i] = row
        
        return lines_cleared

    def select_action(self, board_state, current_piece, training=True):
        """Select best placement using grouped actions"""
        board = board_state["grid"]
        piece = current_piece
        
        placements = self.generate_possible_placements(board, piece)
        
        if not placements:
            return None  # No valid placements (game over)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(placements)
        
        best_value = float('-inf')
        best_placement = None
        
        # Evaluate each possible placement
        for rotation, col in placements:
            result = self.simulate_placement(board, piece, rotation, col)
            if result is None:
                continue
                
            next_board, lines_cleared = result
            
            # Create state representation
            next_state = self.create_state_representation(next_board)
            state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
            
            # Get value from network
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        value = self.model(state_tensor)
                else:
                    value = self.model(state_tensor)
            
            # Add immediate reward for line clears
            immediate_reward = self.calculate_immediate_reward(lines_cleared)
            total_value = immediate_reward + self.gamma * value.item()
            
            if total_value > best_value:
                best_value = total_value
                best_placement = (rotation, col)
        
        return best_placement

    def create_state_representation(self, board):
        """Create 2-channel state representation: board + height map"""
        # Normalize board
        normalized_board = board.astype(np.float32)
        
        # Create height map
        height_map = np.zeros_like(board, dtype=np.float32)
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    height = board.shape[0] - row
                    height_map[:, col] = height / board.shape[0]
                    break
        
        return np.stack([normalized_board, height_map], axis=0)

    def calculate_immediate_reward(self, lines_cleared):
        """Calculate immediate reward based on lines cleared"""
        if lines_cleared == 0:
            return 0.0
        elif lines_cleared == 1:
            return 40.0
        elif lines_cleared == 2:
            return 100.0
        elif lines_cleared == 3:
            return 300.0
        elif lines_cleared == 4:
            return 1200.0  # Tetris!
        else:
            return 0.0

    def remember(self, state, action_info, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Transition(state, action_info, reward, next_state, done))

    def replay(self, episode):
        """Train on a batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.cat(batch.state).to(self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Compute values and loss with mixed precision
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.use_amp):
            # Current state values
            current_values = self.model(state_batch).squeeze()
            
            # Target values
            with torch.no_grad():
                next_values = self.target_model(next_state_batch).squeeze()
                target_values = reward_batch + (1 - done_batch) * self.gamma * next_values
            
            # Compute loss
            loss = F.mse_loss(current_values, target_values)
        
        # Backpropagation with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update target network
        self.soft_update_target_network()
        
        # Decay epsilon
        if episode > 100:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.steps += 1
        
        # Store training stats
        self.training_stats['losses'].append(loss.item())
        avg_value = current_values.mean().item()
        self.training_stats['values'].append(avg_value)
        
        return loss.item()

    def soft_update_target_network(self):
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update_stats(self, episode, total_reward):
        """Update training statistics and return average reward"""
        self.recent_rewards.append(total_reward)
        return np.mean(self.recent_rewards) if self.recent_rewards else 0

    def save(self, filename):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_stats': self.training_stats
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model weights"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.training_stats = checkpoint['training_stats']
        print(f"Model loaded from {filename}")


class TetrisValueNet(nn.Module):
    """Value network for board state evaluation"""
    def __init__(self, input_channels):
        super(TetrisValueNet, self).__init__()
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Single value output
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float()
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        
        # Value prediction
        value = self.value_head(x)
        return value