import pygame
import numpy as np
import cv2
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import os


class StackedFramesDQN(nn.Module):
    """DQN z możliwością przetwarzania kilku klatek jednocześnie"""
    def __init__(self, input_shape, n_actions, n_frames=4):
        super(StackedFramesDQN, self).__init__()
        
        self.n_frames = n_frames
        
        # CNN layers - dostosowane do większej liczby kanałów
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        
        # Oblicz rozmiar po convolution
        conv_out_size = self._get_conv_out((n_frames, input_shape[1], input_shape[2]))
        
        # Dueling DQN architecture
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_out(self, shape):
        """Oblicz rozmiar wyjścia z warstw konwolucyjnych"""
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        # Dueling DQN formula
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(list(self.priorities)[:len(self.buffer)])
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

class ImprovedTetrisAgent:
    def __init__(self, state_shape, n_actions, lr=0.0001, n_frames=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.n_actions = n_actions
        self.n_frames = n_frames
        
        # Sieci neuronowe z Dueling DQN
        self.q_net = StackedFramesDQN(state_shape, n_actions, n_frames).to(self.device)
        self.target_net = StackedFramesDQN(state_shape, n_actions, n_frames).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer z learning rate scheduling
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(50000)
        
        # Frame stacking
        self.frame_stack = deque(maxlen=n_frames)
        
        # Hyperparameters - dostrojone
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Wolniejszy decay
        self.gamma = 0.98
        self.batch_size = 64
        self.target_update = 1000  # Częstsze aktualizacje
        self.steps = 0
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'scores': [],
            'rewards': [],
            'epsilon': [],
            'losses': [],
            'learning_rates': []
        }
        
    def preprocess_state(self, state):
        """Przetwórz stan i dodaj do stack"""
        if len(self.frame_stack) == 0:
            # Wypełnij stack pierwszą klatką
            for _ in range(self.n_frames):
                self.frame_stack.append(state)
        else:
            self.frame_stack.append(state)
            
        return np.stack(self.frame_stack, axis=0)
        
    def reset_frame_stack(self):
        """Reset frame stack na początku epizodu"""
        self.frame_stack.clear()
        
    def remember(self, state, action, reward, next_state, done):
        """Zapisz doświadczenie w prioritized replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
    def act(self, state, training=True):
        """Wybierz akcję używając epsilon-greedy z noise injection"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        # Konwertuj state do tensora
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_tensor)
        
        if training:
            # Dodaj szum do eksploracji
            noise = torch.randn_like(q_values) * 0.1 * self.epsilon
            q_values += noise
            
        return q_values.argmax().item()
        
    def replay(self):
        """Trenuj sieć na batch z prioritized replay buffer"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample z prioritized replay
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Konwertuj do tensorów
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(np.array(weights)).to(self.device)
        
        # Double DQN
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_actions = self.q_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # TD errors dla prioritized replay
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        
        # Weighted loss
        loss = (weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Aktualizuj epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Aktualizuj target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()
        
    def save(self, filename):
        """Zapisz model i statystyki"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_stats': self.training_stats
        }, filename)
        
    def load(self, filename):
        """Wczytaj model i statystyki"""
        checkpoint = torch.load(filename)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
