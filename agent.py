import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import deque, namedtuple
import random
import math

# Define a transition tuple for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ImprovedTetrisAgent:
    def __init__(self, state_shape, n_actions, n_frames=4, use_amp=True):
        # The state_shape should be the shape AFTER frame stacking
        # So if single frame is (2,20,10), with n_frames=4 it becomes (8,20,10)
        self.state_shape = (state_shape[0] * n_frames, *state_shape[1:])
        self.n_actions = n_actions
        self.n_frames = n_frames
        self.use_amp = use_amp
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize frame stack with proper device
        self.frame_stack = deque(maxlen=n_frames)
        self._init_frame_stack()
        
        # Network setup - now uses the full stacked shape
        self.model = TetrisDQN(self.state_shape[0], n_actions).to(self.device)
        self.target_model = TetrisDQN(self.state_shape[0], n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Rest of initialization...
        self.memory = deque(maxlen=200000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.warmup_steps = 2000
        self.tau = 0.005
        
        self.steps = 0
        self.recent_rewards = deque(maxlen=100)
        self.training_stats = {'losses': [], 'q_values': []}
        
        self.scaler = GradScaler(enabled=use_amp)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def _init_frame_stack(self):
        """Initialize frame stack with zeros on correct device"""
        self.frame_stack.clear()
        for _ in range(self.n_frames):
            # Single frame shape is (2,20,10)
            self.frame_stack.append(
                torch.zeros((2, 20, 10), dtype=torch.float32).to(self.device))
    
    def reset_frame_stack(self):
        """Reset frame stack between episodes"""
        self._init_frame_stack()
    
    def preprocess_state(self, state):
        """Convert numpy state to tensor and update frame stack"""
        # Convert to tensor and move to device
        state_tensor = torch.from_numpy(state).float().to(self.device)
        
        # Update frame stack
        self.frame_stack.append(state_tensor)
        
        # Stack frames along channel dimension
        stacked_state = torch.cat(list(self.frame_stack), dim=0)
        return stacked_state.unsqueeze(0)  # Add batch dimension
    
    
    def act(self, state, training=False):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    q_values = self.model(state)
            else:
                q_values = self.model(state)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def replay(self, episode):
        """Train on a batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Compute Q values and loss with mixed precision
        with autocast(enabled=self.use_amp):
            # Current Q values
            current_q = self.model(state_batch).gather(1, action_batch)
            
            # Target Q values
            with torch.no_grad():
                next_q = self.target_model(next_state_batch).max(1)[0]
                target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        # Backpropagation with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update target network
        self.soft_update_target_network()
        
        # Decay epsilon
        if episode > 500:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.steps += 1
        
        # Store training stats
        self.training_stats['losses'].append(loss.item())
        avg_q = current_q.mean().item()
        self.training_stats['q_values'].append(avg_q)
        
        # Log some stats occasionally
        if episode % 100 == 0:
            print(f"Step: {self.steps}, Loss: {loss.item():.4f}, Avg Q: {avg_q:.2f}, Epsilon: {self.epsilon:.4f}")
        
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

class TetrisDQN(nn.Module):
    """Dueling DQN architecture with FP16 support"""
    def __init__(self, input_channels, n_actions):
        super(TetrisDQN, self).__init__()
        
        # Feature extraction - now accepts the full stacked channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Dueling network streams
        self.value_stream = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch, channels, height, width)
        x = x.float()  # Ensure input is float32
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        
        # Dueling streams
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine streams
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals
