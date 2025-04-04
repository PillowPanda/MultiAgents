import numpy as np
import random

class QLearningAgent:
    def __init__(self, name, action_size=4, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.name = name
        self.q_table = {}  # {(state): [q_values for each action]}
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_state_key(self, state):
        # Discretize state into a hashable format
        return tuple(state)

    def choose_action(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return int(np.argmax(self.q_table[key]))

    def learn(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        target = reward + self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
