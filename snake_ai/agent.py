import random
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from game import SnakeGameAI

MAX_MEMORY = 100_000
BATCH_SIZE = 128
ACTION_SPACE_SIZE = 3


class DQNAgent:
    """Class to represent the DQN Agent"""
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.memory = deque(maxlen=MAX_MEMORY)
        self.batch_size = BATCH_SIZE
        self.model = self._build_model()
        self.model_path = './model/model.pth'
        if os.path.isfile(self.model_path):
            self.load_model()
    
    def _build_model(self):
        """Build the DQN model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(11, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss = torch.nn.MSELoss()
        return model

    def remember(self, state, action, reward, next_state, done):
        """Remember the state, action, reward, next state and done flag"""
        self.memory.append((state, action, reward, next_state, done))
    
    def _get_action_vector(self, action_index: int) -> np.ndarray:
        action_vector = np.zeros(ACTION_SPACE_SIZE)
        action_vector[action_index] = 1
        return action_vector
    
    def _get_action_index(self, action_vector: np.ndarray) -> int:
        return np.argmax(action_vector)
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """Choose an action"""
        if np.random.rand() <= self.epsilon:
            return self._get_action_vector(random.randrange(ACTION_SPACE_SIZE))
        act_values = self.model(torch.tensor(state, dtype=torch.float32))
        return self._get_action_vector(np.argmax(act_values.detach().numpy()))
    
    def replay(self):
        """Replay the memory and train the model"""
        if len(self.memory) < self.batch_size:
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure state is 2D
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Ensure next_state is 2D
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)).item()
            
            current_q_values = self.model(state)
            target_f = current_q_values.clone()
            target_f[0, self._get_action_index(action)] = target  # Update the Q-value for the taken action
            
            self.optimizer.zero_grad()
            loss = self.loss(current_q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()
        
    def save_model(self):
        """Save the model"""
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, 'model.pth')
        torch.save(self.model.state_dict(), file_name)
    
    def load_model(self):
        """Load the model from the saved file if it exists."""
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Loaded model from {self.model_path}")

def train_agent():
    game = SnakeGameAI()
    agent = DQNAgent()
    episodes = 2000
    best_score = 0
    n_episodes_to_track = 50
    episode_rewards = deque(maxlen=n_episodes_to_track)
    losses = deque(maxlen=n_episodes_to_track)

    for e in range(episodes):
        game.reset()
        state = game.get_state()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(state)
            reward, done, score = game.play_step(action)
            next_state = game.get_state()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                episode_rewards.append(total_reward)
                print(f'Episode: {e}/{episodes}, Score: {score}, Epsilon: {agent.epsilon}, best_score: {best_score}, avg_reward: {np.mean(episode_rewards)}')
                if score > best_score:
                    best_score = score
                    agent.save_model()
                break
        loss = agent.replay()
        losses.append(loss)
        print(f'Average Loss: {np.mean(losses)}')

# if cmd argument is train, train model, else load the existing model and run    
if __name__ == '__main__':
    train_agent()