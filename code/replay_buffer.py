import numpy as np
import torch

# References:
# 1. https://github.com/ksengin/active-target-localization/blob/master/target_localization/util/replay_buffer.py


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.buffer = []
        self.max_size = int(max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        if self.size > self.max_size:
            del self.buffer[:self.size//5]
            self.size = len(self.buffer)
        
        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        states = torch.zeros((batch_size, self.state_dim))
        actions = torch.zeros((batch_size, 2))
        next_states = torch.zeros((batch_size, self.state_dim))
        rewards = torch.zeros((batch_size, 1))
        dones = torch.zeros((batch_size, 1))
        
        for t, idx in enumerate(indices):
            s, a, r, s_, d = self.buffer[idx]
            states[t] = s
            actions[t] = a
            rewards[t] = r
            next_states[t] = s_
            dones[t] = torch.tensor(d)
        
        return states, actions, rewards, next_states, dones
