"""
OpenAI Gym environment with robots and targets

Environment source code: https://github.com/ksengin/active-target-localization/blob/master/target_localization/envs/tracking_waypoints_env.py
Modifications: Modified the existing environment with number of robots > 1 and the targets moving case
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
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi, cos, sin


# OpenAI gym environment class
class RobotTargetTrackingEnv(gym.GoalEnv):
    
    def __init__(self):
        """
            Init method for the environment
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed()

        self.len_workspace = 20
        self.workspace = np.array([[0, self.len_workspace], [0, self.len_workspace]])
        self.step_size = 0.5
        self.sigma_meas = 1.0
        self.time_step = 1

        self.action_space = spaces.Box(-np.pi, np.pi, shape=(1,), dtype='float32')

        self.dx = 0.1
        self.num_datapts = int(self.len_workspace / self.dx)
        self.x_mesh, self.y_mesh = torch.meshgrid(torch.arange(0, self.len_workspace, self.dx), torch.arange(0, self.len_workspace, self.dx))
        self.xy_mesh = torch.stack((self.x_mesh.reshape(-1), self.y_mesh.reshape(-1))).t()


    def env_parametrization(self, num_targets=1, num_sensors=1, target_motion_omegas=None, meas_model='range', reward_type='heatmap', image_representation=False):
        """ 
            Function for parametrizing the environment
        """
        self.time_step = 1
        self.num_targets = num_targets
        self.x_list = []
        self.y_list = []
        self.true_targets_pos = (torch.rand(self.num_targets, 2) * self.len_workspace)
        self.initial_true_targets_pos = self.true_targets_pos
        self.estimated_targets_mean = self.true_targets_pos
        self.estimated_targets_var = torch.zeros(self.num_targets, 2, 2)
        for index in range(0, self.num_targets):
            self.estimated_targets_var[index] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        self.target_motion_omegas = torch.zeros(self.num_targets, 1)
        if not target_motion_omegas == None:
            self.target_motion_omegas = target_motion_omegas
        else:
            for index in range(0, self.num_targets):
                self.target_motion_omegas[index] = 100

        self.num_sensors = num_sensors
        self.robot_movement_x = []
        self.robot_movement_y = []
        self.sensors_pos = torch.zeros(self.num_sensors, 2)
        for index in range(0, self.num_sensors):
            rand_angle = torch.rand(1) * 2 * np.pi
            self.sensors_pos[index] = self.true_targets_pos.mean(0) + (torch.sqrt(torch.rand(1) + 0.5) * (self.len_workspace / 2) * (torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)])))
            if(self.sensors_pos[index, 0] >= self.len_workspace):
                self.sensors_pos[index, 0] -= (self.sensors_pos[index, 0] - self.len_workspace + 1)
            if(self.sensors_pos[index, 0] <= 0):
                self.sensors_pos[index, 0] = (-self.sensors_pos[index, 0] + 1)
            if(self.sensors_pos[index, 1] >= self.len_workspace):
                self.sensors_pos[index, 1] -= (self.sensors_pos[index, 1] - self.len_workspace + 1)
            if(self.sensors_pos[index, 1] <= 0):
                self.sensors_pos[index, 1] = (-self.sensors_pos[index, 1] + 1)

        self.meas_model = meas_model
        if self.meas_model == 'bearing':
            self.sigma_meas = 0.2
            self.normal_dist_1d_torch = lambda x, mu, sgm: 1.0 / (np.sqrt(2 * np.pi*sgm**2)) * torch.exp(-0.5 / sgm**2 * (np.pi-torch.abs(torch.abs(x-mu) - np.pi))**2)
        else:
            self.sigma_meas = 1.0
            self.normal_dist_1d_torch = lambda x, mu, sgm: 1.0 / (np.sqrt(2 * np.pi*sgm**2)) * np.exp(-0.5 / sgm**2 * (np.abs(x - mu)**2))

        true_obs = self._get_obs()
        self.state = torch.cat((self.sensors_pos[0], torch.tensor(true_obs).float()))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.state.shape, dtype='float32')
        self.reward_type = reward_type

        #self.image_representation = image_representation
        #if image_representation:
        #   self.convnet = ConvNet(out_dim=128, pretrained=False)
        #    self.image = torch.zeros(1,1,256,256)
        #    obs = self.convnet(self.image).squeeze()

    
    def step(self, action):
        """ 
            Function to update the environment
        """
        action = np.clip(action[0], self.action_space.low, self.action_space.high)
        self._set_action(action)

        self.robot_movement_x.append(float(self.sensors_pos[0, 0]))
        self.robot_movement_y.append(float(self.sensors_pos[0, 1]))
        true_obs = self._get_obs()
        self.x_list.append(float(self.true_targets_pos[0, 0]))
        self.y_list.append(float(self.true_targets_pos[0, 1]))
        done = False
        reward = None
        if(self.time_step > 100):
            done = True

        #if self.reward_type == 'heatmap':
        #    self.intersect_heatmaps(obs=obs)
        #reward, done = self.compute_reward()        

        #if self.image_representation:
        #    self.image = torch.flip(self.belief_map.sum(0).t(), dims=(0,))
        #    self.image = F.interpolate(self.image.unsqueeze(0).unsqueeze(0), (256,256), mode='bilinear')
        #    obs = self.convnet(self.image).squeeze()

        self.state = torch.cat((self.sensors_pos[0], true_obs)).detach()
        self.time_step = self.time_step + 1
        self.update_true_targets_pos()
        self.update_estimated_targets_pos()
        return self.state, reward, done, None

    
    def reset(self, **kwargs):
        """ 
            Function to reset the environment
        """
        self.time_step = 1
        self.true_targets_pos = (torch.rand(self.num_targets, 2) * self.len_workspace / 2) + self.len_workspace / 4
        self.x_list = []
        self.y_list = []
        self.estimated_targets_mean = self.true_targets_pos
        self.estimated_targets_var = torch.zeros(self.num_targets, 2, 2)
        for index in range(0, self.num_targets):
            self.estimated_targets_var[index] = torch.tensor([[1, 0], [0, 1]])

        self.sensors_pos = torch.zeros(self.num_sensors, 2)
        self.robot_movement_x = []
        self.robot_movement_y = []
        for index in range(0, self.num_sensors):
            rand_angle = torch.rand(1) * 2 * np.pi
            self.sensors_pos[index] = self.true_targets_pos.mean(0) + (torch.sqrt(torch.rand(1) + 0.5) * (self.len_workspace / 2) * (torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)])))
            if(self.sensors_pos[index, 0] > self.len_workspace):
                self.sensors_pos[index, 0] -= (self.sensors_pos[index, 0] - self.len_workspace + 1)
            if(self.sensors_pos[index, 0] <= 0):
                self.sensors_pos[index, 0] = (-self.sensors_pos[index, 0] + 1)
            if(self.sensors_pos[index, 1] > self.len_workspace):
                self.sensors_pos[index, 1] -= (self.sensors_pos[index, 1] - self.len_workspace + 1)
            if(self.sensors_pos[index, 1] <= 0):
                self.sensors_pos[index, 1] = (-self.sensors_pos[index, 1] + 1)
        true_obs = self._get_obs()       

        #if self.reward_type == 'heatmap':
        #    self.belief_map = torch.ones(self.num_targets, self.num_datapts, self.num_datapts)

        #if self.image_representation:
        #    self.image = torch.zeros(1,1,256,256)
        #    obs = self.convnet(self.image).squeeze()

        self.state = torch.cat((self.sensors_pos[0], true_obs)).detach()
        return self.state

    
    def close(self):
        """ 
            Function to close the environment
        """
        pass

            
    def render(self):
        """ 
            Function for rendering the environment
        """
        plt.cla()
        fig, ax_nstd = plt.subplots(figsize=(8, 8))
        ax_nstd.axvline(c='grey', lw=1)
        ax_nstd.axhline(c='grey', lw=1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, self.len_workspace)
        plt.ylim(0, self.len_workspace)

        #for index in range(0, self.num_targets):
        #    x, y = self.get_correlated_dataset(500, self.estimated_targets_var[index].numpy(), (float(self.estimated_targets_mean[index, 0]), float(self.estimated_targets_mean[index, 1])), (2, 2))
        #    self.confidence_ellipse(x, y, ax_nstd, n_std=1, edgecolor='firebrick')
        #    ax_nstd.scatter(float(self.estimated_targets_mean[index, 0]), float(self.estimated_targets_mean[index, 1]), c='b', s=1)
        #    ax_nstd.legend()
        #    plt.scatter(float(self.true_targets_pos[index, 0]), float(self.true_targets_pos[index, 1]), color='b', marker='*')
        #    plt.scatter([float(self.estimated_targets_mean[index, 0])], [float(self.estimated_targets_mean[index, 1])], color='b', marker="s")
        #    plt.plot([float(self.sensors_pos[0, 0]), float(self.estimated_targets_mean[index, 0])], [float(self.sensors_pos[0, 1]), float(self.estimated_targets_mean[index, 1])], color='r')
        plt.plot(self.robot_movement_x, self.robot_movement_y, 'r--')
        plt.plot(self.x_list, self.y_list, 'b--')
        plt.scatter(float(self.sensors_pos[0, 0]), float(self.sensors_pos[0, 1]), color='r', marker='D') 
        plt.show()


    # reference: https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
    def get_correlated_dataset(self, n, dependency, mu, scale):
        latent = np.random.randn(n, 2)
        dependent = latent.dot(dependency)
        scaled = dependent * scale
        scaled_with_offset = scaled + mu
        return (scaled_with_offset[:, 0], scaled_with_offset[:, 1])


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

            if self.image_representation:
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
        self.belief_map /= self.belief_map.max(-1)[0].max(-1)[0].reshape(-1, 1, 1)


    # reference: https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
    def confidence_ellipse(self, x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
           Inputs:
           x: the x-coordinate datapoints
           y: the y-coordinate datapoints
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
        # Using a special case to obtain the eigenvalues of this two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

        # Calculating the stdandard deviation of x from the squareroot of the variance and multiplying with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)  


    def _reset_sim(self, **kwargs):
        raise NotImplementedError()


    def _get_obs(self):
        """ 
            Get observation function
            Returns the noisy relative measurement depending on the measurement model (bearing or range)
        """
        targets_pos = self.true_targets_pos
        sensor_pos = self.sensors_pos[0]

        if self.meas_model == 'bearing':
            true_measurement = torch.atan2(targets_pos[:, 1] - sensor_pos[1], targets_pos[:, 0] - sensor_pos[0])
        elif self.meas_model == 'range':
            true_measurement = torch.norm(targets_pos - sensor_pos, p=2, dim=1)

        true_obs = true_measurement + (self.sigma_meas * torch.randn(targets_pos.shape[0]))
        return true_obs


    def update_true_targets_pos(self):
        """
            Function to update the true target positions when time step increases (assuming circular target motions)
        """
        for index in range(0, self.num_targets):
            self.true_targets_pos[index] = torch.tensor([2*np.cos((self.time_step - 1) / float(self.target_motion_omegas[index])) + float(self.initial_true_targets_pos[index, 0]) - 2.0, 2*np.sin((self.time_step - 1) / float(self.target_motion_omegas[index])) + float(self.initial_true_targets_pos[index, 1])])
        

    def update_estimated_targets_pos(self):
        """
            Function to update the estimated target positions when time step increases (using ekf)
        """
        true_obs = self._get_obs()
        for index in range(0, self.num_targets):
            z_true = np.zeros((1, 1))
            z_true[0, 0] = true_obs[index]
            q_matrix = 0.2 * np.eye(2)
            x_matrix = np.array([[float(self.estimated_targets_mean[index, 0])], [float(self.estimated_targets_mean[index, 1])]])
            sigma_matrix = self.estimated_targets_var[index].numpy() + q_matrix

            z_pred = np.zeros((1, 1))
            h_matrix = np.zeros((1, 2))
            for index in range(0, 1):
                z_pred[index][0] = np.linalg.norm([[x_matrix[0, 0] - float(self.sensors_pos[0, 0])], [x_matrix[1, 0] - float(self.sensors_pos[0, 1])]], 2)
                h_matrix[index, 0] = (-1.0 / z_pred[index]) * (float(self.sensors_pos[0, 0]) - x_matrix[0, 0])
                h_matrix[index, 1] = (-1.0 / z_pred[index]) * (float(self.sensors_pos[0, 1]) - x_matrix[1, 0])
            
            res = (z_true - z_pred)
            r_matrix = 1.0 * 1.0 * np.eye(1)
            s_matrix = np.matmul(np.matmul(h_matrix, sigma_matrix), h_matrix.T) + r_matrix
            k_matrix = np.matmul(np.matmul(sigma_matrix, h_matrix.T), np.linalg.inv(s_matrix))

            x_matrix_tplus1 = x_matrix + (np.matmul(k_matrix, res))
            sigma_matrix_tplus1 = np.matmul(np.matmul((np.eye(2) - np.matmul(k_matrix, h_matrix)), sigma_matrix), (np.eye(2) - np.matmul(k_matrix, h_matrix)).T) + np.matmul(np.matmul(k_matrix, r_matrix), k_matrix.T)
            target_xhat_tplus1 = x_matrix_tplus1[0, 0]
            target_yhat_tplus1 = x_matrix_tplus1[1, 0]
            
            self.estimated_targets_mean[index, 0] = target_xhat_tplus1
            self.estimated_targets_mean[index, 1] = target_yhat_tplus1
            self.estimated_targets_var[index] = torch.tensor(sigma_matrix_tplus1)


    def _get_true_target_position(self):
        """
            Function to return the true target positions.
        """
        return self.true_targets_pos


    def _get_estimated_target_position(self):
        """
            Function to return the estimated target positions.
        """
        return self.estimated_targets_pos

    
    def get_posterior_map(self):
        return self.posterior_map

    
    def _set_action(self, action):
        """
            Applies the given action to the sensor.
        """
        action = torch.tensor(action).float()
        vector = self.step_size * torch.tensor([torch.cos(action), torch.sin(action)])
        sensor_pos = self.sensors_pos[0] + vector
        if(sensor_pos[0] >= self.len_workspace):
            sensor_pos[0] = self.sensors_pos[0, 0]
        if(sensor_pos[1] >= self.len_workspace):
            sensor_pos[1] = self.sensors_pos[0, 1]
        self.sensors_pos[0] = sensor_pos   

        
    def seed(self, seed=None):
        """
            Function that returns a random seed using OpenAI Gym seeding.
        """
        _, seed = seeding.np_random(seed)
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
