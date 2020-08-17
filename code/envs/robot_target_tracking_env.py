"""
Environment source code: https://github.com/ksengin/active-target-localization/blob/master/target_localization/envs/tracking_waypoints_env.py
OpenAI Gym environment with robots and targets
"""

# header files
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

import gym
from gym import error, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# env class
class RobotTargetTrackingEnv(gym.GoalEnv):
    
    def __init__(self):
        """
            Init method for the environment
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed()

        self.len_workspace = 20
        len_workspace = self.len_workspace
        self.workspace = np.array([[0, len_workspace], [0, len_workspace]])
        self.step_size = 0.5
        self.sigma_meas = 1.0

        n_actions = 1
        self.action_space = spaces.Box(-np.pi, np.pi, shape=(n_actions,), dtype='float32')

        self.dx = 0.1
        self.num_datapts = int(self.len_workspace / self.dx)
        self.x_mesh, self.y_mesh = torch.meshgrid(torch.arange(0, len_workspace, self.dx), torch.arange(0, len_workspace, self.dx))
        self.xy_mesh = torch.stack((self.x_mesh.reshape(-1), self.y_mesh.reshape(-1))).t()


    def env_parametrization(self, num_targets:int=1, reward_type:str='fim', image_representation:bool=False, vis=None, meas_model='range', static_target=True, augment_state=True, im_loss=True):
        """ 
            Function for parametrizing the environment
        """
        self.num_targets = num_targets
        self.target_pos = (torch.rand(self.num_targets, 2) * self.len_workspace)
        rand_angle = torch.rand(1)*2*np.pi
        self.sensor_pos = self.target_pos.mean(0) + torch.sqrt(torch.rand(1)+0.5)*self.len_workspace/2 * torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)])

        self.augment_state = augment_state
        self.meas_model = meas_model
        if self.meas_model == 'bearing':
            self.sigma_meas = 0.2

        if self.meas_model == 'bearing':
            self.normal_dist_1d_torch = lambda x, mu, sgm: 1 / (np.sqrt(2*np.pi*sgm**2)) * torch.exp(-0.5/sgm**2 * (np.pi-torch.abs(torch.abs(x-mu)-np.pi))**2)
        else:
            self.normal_dist_1d_torch = lambda x, mu, sgm: 1 / (np.sqrt(2*np.pi*sgm**2)) * np.exp(-0.5/sgm**2 * (np.abs(x-mu)**2))


        obs = self._get_obs()

        self.reward_type = reward_type
        plt.figure(figsize=(8, 8))

        self.image_representation = image_representation
        self.im_loss = im_loss
        if image_representation:
            self.vis = vis[0] if vis is not None else visdom.Visdom()
            self.vis_env_name = vis[1] if vis is not None else 'env_image'
            self.convnet = ConvNet(out_dim=128, pretrained=False)
            self.image = torch.zeros(1,1,256,256)
            obs = self.convnet(self.image).squeeze()

        # self position and bearing measurement
        self.state = torch.cat((self.sensor_pos, torch.tensor(obs).float()))
        if self.augment_state:
            self.predictions_flat = torch.rand_like(self.target_pos).reshape(-1)
            self.state = torch.cat((self.state, self.predictions_flat))

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.state.shape, dtype='float32')
        self.static_target = static_target

    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        # bearing measurement
        obs = self._get_obs()
        done = False

        if self.reward_type == 'heatmap':
            self.intersect_heatmaps(obs=obs)
        
        if self.image_representation:
            self.image = torch.flip(self.belief_map.sum(0).t(), dims=(0,))
            self.image = F.interpolate(self.image.unsqueeze(0).unsqueeze(0), (256,256), mode='bilinear')
            obs = self.convnet(self.image).squeeze()

            # self.vis.image(self.image.squeeze(), win='image', env=self.vis_env_name)

        reward, done = self.compute_reward()

        self.state = torch.cat((self.sensor_pos, obs)).detach()
        if self.augment_state:
            self.predictions_flat = self.predictions.reshape(-1)
            self.state = torch.cat((self.state, self.predictions_flat)).detach()
        if not self.static_target:
            self.target_pos = brownian_one_pass(self.target_pos, delta=0.25)

        return self.state, reward, done, None

    
    def reset(self, **kwargs):
        len_ws = self.len_workspace
        self.target_pos = torch.rand(self.num_targets, 2) * len_ws/2 + len_ws/4
        rand_angle = torch.rand(1)*2*np.pi
        self.sensor_pos = self.target_pos.mean(0) + torch.sqrt(torch.rand(1)+0.5)*len_ws/2 * torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)])

        obs = self._get_obs()

        if self.reward_type == 'fim':
            self.info_acc = InformationAccumulator(init_sensor_loc=self.sensor_pos, target_loc=self.target_pos)
        elif self.reward_type == 'ci':
            self.mean_belief = self.len_workspace/2 * np.ones(2)
            self.cov_belief = self.len_workspace * np.eye(2) # essentially a uniform prior
        elif self.reward_type == 'heatmap':
            self.belief_map = torch.ones(self.num_targets, self.num_datapts, self.num_datapts)
        
        if self.image_representation:
            self.image = torch.zeros(1,1,256,256)
            obs = self.convnet(self.image).squeeze()

        self.predictions_flat = torch.rand_like(self.target_pos).reshape(-1)
        self.state = torch.cat((self.sensor_pos, obs)).detach()
        if self.augment_state:
            self.state = torch.cat((self.state, self.predictions_flat))
        self.state_hist = self.sensor_pos.reshape(1,-1) # can be used to compute cumulative FIM determinant
        return self.state

    
    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

            
    def render(self):
        plt.cla()
        plt.contourf(self.x_mesh.cpu().detach().numpy(), self.y_mesh.cpu().detach().numpy(), self.belief_map.sum(0).cpu().detach().numpy(), cmap=cm.inferno)
        plt.plot(self.state_hist[:,0], self.state_hist[:,1], 's', c='r')
        plt.plot(self.state[0], self.state[1], 's', c=np.array([1,0.3,0.3]))
        plt.plot(self.target_pos[:,0], self.target_pos[:,1], 'o', c='b')
        if self.reward_type == 'heatmap':
            plt.plot(self.predictions[:,0], self.predictions[:,1], 'o', c=np.array([1,1,0]))
        plt.axis('equal')
        plt.pause(0.001)


    def compute_reward(self, sparse:bool=False):
        """ Function for computing the reward
        Type 'fim' is used in preliminary experiments
        Type 'heatmap' computes the mean localization error between predicted and true target positions
        """
        true_target_pos = self.target_pos
        sensor_pos_all = self.state_hist
        done = False
        reward = 0.

        if self.reward_type == 'fim' and self.info_acc is not None:
            reward = self.info_acc.fisher_determinant(self.sensor_pos)
            reward /= self.num_targets
        elif self.reward_type == 'fim':
            for target_pos in true_target_pos:
                vecs_to_target = target_pos - sensor_pos_all
                angles = torch.atan2(vecs_to_target[:,1], vecs_to_target[:,0])
                dists = torch.norm(vecs_to_target, p=2, dim=1)

                for idx in range(sensor_pos_all.shape[0] - 1):
                    for jdx in range(idx+1, sensor_pos_all.shape[0]):
                        reward += torch.sin(angles[jdx] - angles[idx])**2 / (dists[idx]**2 * dists[jdx]**2)

        elif self.reward_type == 'heatmap': # argmax on intersected heatmaps
            self.predictions = torch.empty(self.num_targets, 2)
            for idx in range(self.num_targets):
                self.predictions[idx] = (self.belief_map[idx] == self.belief_map[idx].max()).nonzero().float().mean(0) * self.dx

            if self.im_loss and self.image_representation:
                done = False if self.belief_map.sum() > 10 else True
                return - self.image.mean(), done

            distances = torch.norm(true_target_pos - self.predictions, p=2, dim=1)
            distance = distances.mean()
            if sparse:
                reward = torch.tensor(-1)
            else:
                reward = torch.tensor(-distance)
            
            # episode ends if error mean is below a threshold
            if distance < self.len_workspace/100:
                reward = torch.tensor(100) if sparse else reward
                done = True
        
        return reward, done

    
    def intersect_heatmaps(self, obs):
        """ Function for updating the belief map
        Computes the probability of each bin containing the target position (one channel for each target)
        The belief map posterior is computed by incorporating the new measurement to the prior map
        """
        ws_vectors = self.xy_mesh - self.sensor_pos
        if self.meas_model == 'bearing':
            ws_measurements = np.arctan2(ws_vectors[:,1], ws_vectors[:,0])
        elif self.meas_model == 'range':
            ws_measurements = torch.norm(ws_vectors, p=2, dim=1)

        measurement_likelihood = torch.stack([(self.normal_dist_1d_torch(ws_measurements, obs[i], self.sigma_meas)) for i in range(self.num_targets)])
        measurement_likelihood = measurement_likelihood.reshape(self.num_targets, -1, self.num_datapts)

        self.belief_map = self.belief_map * measurement_likelihood
        self.belief_map /= self.belief_map.max(-1)[0].max(-1)[0].reshape(-1,1,1)


    def _reset_sim(self, **kwargs):
        raise NotImplementedError()


    def _get_obs(self):
        """ Get observation function
        Returns the noisy relative measurement depending on the measurement model (bearing or range)
        """
        target_pos = self.target_pos
        cur_pos = self.sensor_pos

        if self.meas_model == 'bearing':
            true_measurement = torch.atan2(target_pos[:,1]-cur_pos[1], target_pos[:,0]-cur_pos[0])
        elif self.meas_model == 'range':
            true_measurement = torch.norm(target_pos-cur_pos, p=2, dim=1)

        obs = true_measurement + self.sigma_meas * torch.randn(target_pos.shape[0])

        return obs


    def _get_true_target_position(self):
        return self.target_pos

    
    def get_posterior_map(self):
        return self.posterior_map

    
    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        action = torch.tensor(action).float()
        vector = self.step_size * torch.tensor([torch.cos(action), torch.sin(action)])

        self.sensor_pos = self.sensor_pos + vector
        self.state_hist = torch.cat((self.state_hist, self.sensor_pos.reshape(-1,2)))

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        
    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError()

        
    def _sample_goal(self):
        return torch.rand(2)

    
    def _env_setup(self, initial_qpos):
        pass

    
    def _viewer_setup(self):
        pass

    
    def _render_callback(self):
        pass

    
    def _step_callback(self):
        pass
