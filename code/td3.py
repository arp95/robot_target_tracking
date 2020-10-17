import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# References:
# 1. https://github.com/sfujim/TD3/blob/master/TD3.py
# 2. https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2/blob/master/TD3.py


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        a[:, 0] = torch.tanh(a[:, 0]) * self.max_action
        a[:, 1] = torch.sigmoid(a[:, 1])
        a[:, 2] = torch.tanh(a[:, 2]) * self.max_action
        a[:, 3] = torch.sigmoid(a[:, 3])
        return a
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, action_dim)
        
    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class TD3(object):
    def __init__(self, lr, state_dim, action_dim, max_action, joint_critic_optim: bool = False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.joint_critic_optim = joint_critic_optim
        if joint_critic_optim:
            self.critic_optim = optim.Adam(chain(self.critic_1.parameters(), self.critic_2.parameters()), lr=lr)

        self.max_action = max_action
        self.critic_loss = F.mse_loss
    
    def select_action(self, state):
        state = state.reshape(1, -1).to(self.device)
        return self.actor(state).cpu().data.flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, tau, policy_noise, policy_delay):
        
        for idx in range(n_iter):
            # Sample a batch of transitions from the replay buffer
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = state.to(self.device)
            action = action_.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
            
            # Select next action according to target policy
            noise = torch.normal(0, policy_noise, action.shape).to(self.device)
            next_action = (self.actor_target(next_state) + noise)
            next_action[:, 0] = next_action[:, 0].clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # Optimize the critic networks
            if not self.joint_critic_optim:
                current_Q1 = self.critic_1(state, action)
                loss_Q1 = self.critic_loss(current_Q1, target_Q)
                self.critic_1_optimizer.zero_grad()
                loss_Q1.backward()
                self.critic_1_optimizer.step()
                
                current_Q2 = self.critic_2(state, action)
                loss_Q2 = self.critic_loss(current_Q2, target_Q)
                self.critic_2_optimizer.zero_grad()
                loss_Q2.backward()
                self.critic_2_optimizer.step()
            
            else: # optimization as in the original implementation
                current_Q1 = self.critic_1(state, action)
                current_Q2 = self.critic_2(state, action)
                loss_Q = self.critic_loss(current_Q1, target_Q) + self.critic_loss(current_Q2, target_Q)
                self.critic_optim.zero_grad()
                loss_Q.backward()
                self.critic_optim.step()

            # Delayed policy updates
            if idx % policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * target_param.data + (1-tau) * param.data)
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(tau * target_param.data + (1-tau) * param.data)
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(tau * target_param.data + (1-tau) * param.data)
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), f'{directory}/{name}_actor.pth')
        torch.save(self.actor_target.state_dict(), f'{directory}/{name}_actor_target.pth')
        
        torch.save(self.critic_1.state_dict(), f'{directory}/{name}_crtic_1.pth')
        torch.save(self.critic_1_target.state_dict(), f'{directory}/{name}_critic_1_target.pth')
        
        torch.save(self.critic_2.state_dict(), f'{directory}/{name}_crtic_2.pth')
        torch.save(self.critic_2_target.state_dict(), f'{directory}/{name}_critic_2_target.pth')
        
    def load(self, directory, name):
        self.load_actor(directory, name)
        
        self.critic_1.load_state_dict(torch.load(f'{directory}/{name}_crtic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(f'{directory}/{name}_critic_1_target.pth'))
        
        self.critic_2.load_state_dict(torch.load(f'{directory}/{name}_crtic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(f'{directory}/{name}_critic_2_target.pth'))
        
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load(f'{directory}/{name}_actor.pth'))
        self.actor_target.load_state_dict(torch.load(f'{directory}/{name}_actor_target.pth'))
