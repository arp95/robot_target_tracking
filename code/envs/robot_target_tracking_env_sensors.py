"""
OpenAI Gym environment with robots and targets

Environment source code: https://github.com/ksengin/active-target-localization/blob/master/target_localization/envs/tracking_waypoints_env.py
Modifications: Modified the existing environment with number of robots > 1 and the targets moving case
"""


# header files
import os
import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import gym
from gym import error, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from math import pi, cos, sin
from scipy.stats import multivariate_normal
from convnet import *


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
        self.sigma_meas = 1.0
        self.time_step = 1

        self.action_space = spaces.Box(-np.pi, np.pi, shape=(1,), dtype='float32')
        self.convnet = ConvNet()


    def env_parametrization(self, num_targets=2, num_sensors=2, target_motion_omegas=None, meas_model='range', include_cnn=True):
        """  
            Function for parametrizing the environment
        """
        self.include_cnn = include_cnn
        self.x1_list = []
        self.y1_list = []
        self.x2_list = []
        self.y2_list = []
        self.x3_list = []
        self.y3_list = []
        self.x4_list = []
        self.y4_list = []
        self.time_step = 1
        self.num_targets = num_targets
        self.true_targets_radii = torch.rand(self.num_targets)*5.0+2.0
        self.true_targets_pos = (torch.rand(self.num_targets, 2)*self.len_workspace)
        self.initial_true_targets_pos = self.true_targets_pos.clone()
        self.estimated_targets_mean = self.true_targets_pos.clone()
        self.estimated_targets_var = torch.zeros(self.num_targets, 2, 2)
        for index in range(0, self.num_targets):
            self.estimated_targets_var[index] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.target_motion_omegas = torch.rand(self.num_targets)*100.0

        self.heatmap = torch.zeros(self.len_workspace, self.len_workspace)
        for index in range(0, self.num_targets):
            x = np.linspace(0, self.len_workspace, self.len_workspace)
            y = np.linspace(0, self.len_workspace, self.len_workspace)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal(self.estimated_targets_mean[index], self.estimated_targets_var[index])
            self.heatmap += rv.pdf(pos)
        if include_cnn:
            image = F.interpolate(self.heatmap.unsqueeze(0).unsqueeze(0), (256, 256), mode='bilinear', align_corners=True)
            image = image.float()
            true_obs = self.convnet(image).squeeze()
        else:  
            true_obs = self.heatmap.flatten()

        self.robot_movement_x_1 = []
        self.robot_movement_y_1 = []
        self.robot_movement_x_2 = []
        self.robot_movement_y_2 = []
        self.robot_movement_x_3 = []
        self.robot_movement_y_3 = []
        self.robot_movement_x_4 = []
        self.robot_movement_y_4 = []
        self.num_sensors = num_sensors
        self.sensors_pos = torch.zeros(self.num_sensors, 2)
        for index in range(0, self.num_sensors):
            rand_angle = torch.rand(1)*2*np.pi
            self.sensors_pos[index] = self.true_targets_pos.mean(0) + (torch.sqrt(torch.rand(1)+0.5)*(self.len_workspace/2)*(torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)])))
            if(self.sensors_pos[index, 0]>=self.len_workspace):
                self.sensors_pos[index, 0] -= (self.sensors_pos[index, 0]-self.len_workspace+1)
            if(self.sensors_pos[index, 0]<=0):
                self.sensors_pos[index, 0] = (-self.sensors_pos[index, 0]+1)
            if(self.sensors_pos[index, 1]>=self.len_workspace):
                self.sensors_pos[index, 1] -= (self.sensors_pos[index, 1]-self.len_workspace+1)
            if(self.sensors_pos[index, 1]<=0):
                self.sensors_pos[index, 1] = (-self.sensors_pos[index, 1]+1)

        self.robot_movement_x_1.append(float(self.sensors_pos[0, 0]))
        self.robot_movement_y_1.append(float(self.sensors_pos[0, 1]))
        self.robot_movement_x_2.append(float(self.sensors_pos[1, 0]))
        self.robot_movement_y_2.append(float(self.sensors_pos[1, 1]))
        #self.robot_movement_x_3.append(float(self.sensors_pos[2, 0]))
        #self.robot_movement_y_3.append(float(self.sensors_pos[2, 1]))
        #self.robot_movement_x_4.append(float(self.sensors_pos[3, 0]))
        #self.robot_movement_y_4.append(float(self.sensors_pos[3, 1]))
        self.x1_list.append(float(self.true_targets_pos[0, 0]))
        self.y1_list.append(float(self.true_targets_pos[0, 1]))
        self.x2_list.append(float(self.true_targets_pos[1, 0]))
        self.y2_list.append(float(self.true_targets_pos[1, 1]))
        #self.x3_list.append(float(self.true_targets_pos[2, 0]))
        #self.y3_list.append(float(self.true_targets_pos[2, 1]))
        #self.x4_list.append(float(self.true_targets_pos[3, 0]))
        #self.y4_list.append(float(self.true_targets_pos[3, 1]))

        self.meas_model = meas_model
        if self.meas_model == 'bearing':
            self.sigma_meas = 0.2
            self.normal_dist_1d_torch = lambda x, mu, sgm: 1.0/(np.sqrt(2*np.pi*sgm**2))*torch.exp(-0.5/sgm**2*(np.pi-torch.abs(torch.abs(x-mu) - np.pi))**2)
        else:
            self.sigma_meas = 1.0
            self.normal_dist_1d_torch = lambda x, mu, sgm: 1.0/(np.sqrt(2*np.pi*sgm**2))*np.exp(-0.5/sgm**2*(np.abs(x-mu)**2))
        self.state = torch.cat((self.sensors_pos[0], self.sensors_pos[1], torch.tensor(true_obs).float()))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.state.shape, dtype='float32')

    
    def step(self, action, step_size):
        """ 
            Function to update the environment
        """
        start = time.time()
        self._set_action(action, step_size)

        self.time_step = self.time_step + 1
        self.update_true_targets_pos()
        self.update_estimated_targets_pos()

        self.robot_movement_x_1.append(float(self.sensors_pos[0, 0]))
        self.robot_movement_y_1.append(float(self.sensors_pos[0, 1]))
        self.robot_movement_x_2.append(float(self.sensors_pos[1, 0]))
        self.robot_movement_y_2.append(float(self.sensors_pos[1, 1]))
        #self.robot_movement_x_3.append(float(self.sensors_pos[2, 0]))
        #self.robot_movement_y_3.append(float(self.sensors_pos[2, 1]))
        #self.robot_movement_x_4.append(float(self.sensors_pos[3, 0]))
        #self.robot_movement_y_4.append(float(self.sensors_pos[3, 1]))
        self.x1_list.append(float(self.true_targets_pos[0, 0]))
        self.y1_list.append(float(self.true_targets_pos[0, 1]))
        self.x2_list.append(float(self.true_targets_pos[1, 0]))
        self.y2_list.append(float(self.true_targets_pos[1, 1]))
        #self.x3_list.append(float(self.true_targets_pos[2, 0]))
        #self.y3_list.append(float(self.true_targets_pos[2, 1]))
        #self.x4_list.append(float(self.true_targets_pos[3, 0]))
        #self.y4_list.append(float(self.true_targets_pos[3, 1]))

        self.heatmap = torch.zeros(self.len_workspace, self.len_workspace)
        for index in range(0, self.num_targets):
            x = np.linspace(0, self.len_workspace, self.len_workspace)
            y = np.linspace(0, self.len_workspace, self.len_workspace)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal(self.estimated_targets_mean[index], self.estimated_targets_var[index])
            self.heatmap += rv.pdf(pos)
        if self.include_cnn:
            image = F.interpolate(self.heatmap.unsqueeze(0).unsqueeze(0), (256, 256), mode='bilinear', align_corners=True)
            image = image.float()
            true_obs = self.convnet(image).squeeze()
        else:
            true_obs = self.heatmap.flatten()

        done = False
        reward = None
        reward, done = self.compute_reward()
        for index in range(0, self.num_sensors):
            if(self.time_step>200 or float(self.sensors_pos[index, 0]) <= 0 or float(self.sensors_pos[index, 1]) <= 0 or float(self.sensors_pos[index, 0]) >= self.len_workspace or float(self.sensors_pos[index, 1]) >= self.len_workspace):
                done = True

        self.state = torch.cat((self.sensors_pos[0], self.sensors_pos[1], torch.tensor(true_obs).float()))
        end = time.time() - start
        return self.state, reward, done, None, self.estimated_targets_var

    
    def reset(self, **kwargs):
        """ 
            Function to reset the environment
        """
        self.x1_list = []
        self.y1_list = []
        self.x2_list = []
        self.y2_list = []
        self.x3_list = []
        self.y3_list = []
        self.x4_list = []
        self.y4_list = []
        self.time_step = 1
        self.true_targets_radii = torch.rand(self.num_targets)*5.0+2.0
        self.true_targets_pos = (torch.rand(self.num_targets, 2)*self.len_workspace)
        self.initial_true_targets_pos = self.true_targets_pos.clone()
        self.estimated_targets_mean = self.true_targets_pos.clone()
        self.estimated_targets_var = torch.zeros(self.num_targets, 2, 2)
        for index in range(0, self.num_targets):
            self.estimated_targets_var[index] = torch.tensor([[1, 0], [0, 1]])
        self.target_motion_omegas = torch.rand(self.num_targets)*100.0

        self.robot_movement_x_1 = []
        self.robot_movement_y_1 = []
        self.robot_movement_x_2 = []
        self.robot_movement_y_2 = []
        self.robot_movement_x_3 = []
        self.robot_movement_y_3 = []
        self.robot_movement_x_4 = []
        self.robot_movement_y_4 = []
        self.sensors_pos = torch.zeros(self.num_sensors, 2)
        for index in range(0, self.num_sensors):
            rand_angle = torch.rand(1)*2*np.pi
            self.sensors_pos[index] = self.true_targets_pos.mean(0)+(torch.sqrt(torch.rand(1)+0.5)*(self.len_workspace/2)*(torch.tensor([torch.cos(rand_angle), torch.sin(rand_angle)])))
            if(self.sensors_pos[index, 0]>=self.len_workspace):
                self.sensors_pos[index, 0] -= (self.sensors_pos[index, 0]-self.len_workspace+1)
            if(self.sensors_pos[index, 0]<=0):
                self.sensors_pos[index, 0] = (-self.sensors_pos[index, 0]+1)
            if(self.sensors_pos[index, 1]>=self.len_workspace):
                self.sensors_pos[index, 1] -= (self.sensors_pos[index, 1]-self.len_workspace+1)
            if(self.sensors_pos[index, 1]<=0):
                self.sensors_pos[index, 1] = (-self.sensors_pos[index, 1]+1)  

        self.robot_movement_x_1.append(float(self.sensors_pos[0, 0]))
        self.robot_movement_y_1.append(float(self.sensors_pos[0, 1]))
        self.robot_movement_x_2.append(float(self.sensors_pos[1, 0]))
        self.robot_movement_y_2.append(float(self.sensors_pos[1, 1]))
        #self.robot_movement_x_3.append(float(self.sensors_pos[2, 0]))
        #self.robot_movement_y_3.append(float(self.sensors_pos[2, 1]))
        #self.robot_movement_x_4.append(float(self.sensors_pos[3, 0]))
        #self.robot_movement_y_4.append(float(self.sensors_pos[3, 1]))
        self.x1_list.append(float(self.true_targets_pos[0, 0]))
        self.y1_list.append(float(self.true_targets_pos[0, 1]))
        self.x2_list.append(float(self.true_targets_pos[1, 0]))
        self.y2_list.append(float(self.true_targets_pos[1, 1]))
        #self.x3_list.append(float(self.true_targets_pos[2, 0]))
        #self.y3_list.append(float(self.true_targets_pos[2, 1]))
        #self.x4_list.append(float(self.true_targets_pos[3, 0]))
        #self.y4_list.append(float(self.true_targets_pos[3, 1]))

        self.heatmap = torch.zeros(self.len_workspace, self.len_workspace)
        for index in range(0, self.num_targets):
            x = np.linspace(0, self.len_workspace, self.len_workspace)
            y = np.linspace(0, self.len_workspace, self.len_workspace)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal(self.estimated_targets_mean[index], self.estimated_targets_var[index])
            self.heatmap += rv.pdf(pos)
        if self.include_cnn:
            image = F.interpolate(self.heatmap.unsqueeze(0).unsqueeze(0), (256, 256), mode='bilinear', align_corners=True)
            image = image.float()
            true_obs = self.convnet(image).squeeze()
        else:
            true_obs = self.heatmap.flatten()

        self.state = torch.cat((self.sensors_pos[0], self.sensors_pos[1], torch.tensor(true_obs).float()))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.state.shape, dtype='float32')
        return self.state, self.sensors_pos, self.estimated_targets_mean, self.true_targets_radii, self.target_motion_omegas


    def set_env(self, sensors_pos, estimated_targets_mean, true_targets_radii, target_motion_omegas):
        self.x1_list = []
        self.y1_list = []
        self.x2_list = []
        self.y2_list = []
        self.x3_list = []
        self.y3_list = []
        self.x4_list = []
        self.y4_list = []
        self.time_step = 1
        self.true_targets_radii = true_targets_radii
        self.true_targets_pos = estimated_targets_mean
        self.initial_true_targets_pos = self.true_targets_pos.clone()
        self.estimated_targets_mean = self.true_targets_pos.clone()
        self.estimated_targets_var = torch.zeros(self.num_targets, 2, 2)
        for index in range(0, self.num_targets):
            self.estimated_targets_var[index] = torch.tensor([[1, 0], [0, 1]])
        self.target_motion_omegas = target_motion_omegas
        
        self.robot_movement_x_1 = []
        self.robot_movement_y_1 = []
        self.robot_movement_x_2 = []
        self.robot_movement_y_2 = []
        self.robot_movement_x_3 = []
        self.robot_movement_y_3 = []
        self.robot_movement_x_4 = []
        self.robot_movement_y_4 = []
        self.sensors_pos = sensors_pos

        self.robot_movement_x_1.append(float(self.sensors_pos[0, 0]))
        self.robot_movement_y_1.append(float(self.sensors_pos[0, 1]))
        self.robot_movement_x_2.append(float(self.sensors_pos[1, 0]))
        self.robot_movement_y_2.append(float(self.sensors_pos[1, 1]))
        #self.robot_movement_x_3.append(float(self.sensors_pos[2, 0]))
        #self.robot_movement_y_3.append(float(self.sensors_pos[2, 1]))
        #self.robot_movement_x_4.append(float(self.sensors_pos[3, 0]))
        #self.robot_movement_y_4.append(float(self.sensors_pos[3, 1]))
        self.x1_list.append(float(self.true_targets_pos[0, 0]))
        self.y1_list.append(float(self.true_targets_pos[0, 1]))
        self.x2_list.append(float(self.true_targets_pos[1, 0]))
        self.y2_list.append(float(self.true_targets_pos[1, 1]))
        #self.x3_list.append(float(self.true_targets_pos[2, 0]))
        #self.y3_list.append(float(self.true_targets_pos[2, 1]))
        #self.x4_list.append(float(self.true_targets_pos[3, 0]))
        #self.y4_list.append(float(self.true_targets_pos[3, 1]))

        self.heatmap = torch.zeros(self.len_workspace, self.len_workspace)
        for index in range(0, self.num_targets):
            x = np.linspace(0, self.len_workspace, self.len_workspace)
            y = np.linspace(0, self.len_workspace, self.len_workspace)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal(self.estimated_targets_mean[index], self.estimated_targets_var[index])
            self.heatmap += rv.pdf(pos)
        if self.include_cnn:
            image = F.interpolate(self.heatmap.unsqueeze(0).unsqueeze(0), (256, 256), mode='bilinear', align_corners=True)
            image = image.float()
            true_obs = self.convnet(image).squeeze()
        else:
            true_obs = self.heatmap.flatten()

        self.state = torch.cat((self.sensors_pos[0], self.sensors_pos[1], torch.tensor(true_obs).float()))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.state.shape, dtype='float32')
        return self.state, self.sensors_pos, self.estimated_targets_mean, self.true_targets_radii, self.target_motion_omegas

    
    def close(self):
        """ 
            Function to close the environment
        """
        pass

            
    def render(self):
        """ 
            Function for rendering the environment
        """
        heatmap = torch.zeros(256, 256)
        for index in range(0, self.num_targets):
            x = np.linspace(0, self.len_workspace, 256)
            y = np.linspace(0, self.len_workspace, 256)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal(self.estimated_targets_mean[index], self.estimated_targets_var[index])
            heatmap += rv.pdf(pos)
        x = np.linspace(0, self.len_workspace, 256)
        y = np.linspace(0, self.len_workspace, 256)
        X, Y = np.meshgrid(x, y)
        plt.cla()
        plt.title("Time step = " + str(self.time_step))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, self.len_workspace)
        plt.ylim(0, self.len_workspace)
        plt.contourf(X, Y, heatmap, cmap=cm.inferno)
        plt.plot(self.x1_list, self.y1_list, 'b--')
        plt.plot(self.x1_list[len(self.x1_list)-1], self.y1_list[len(self.y1_list)-1], 'o', c='b', marker='*')
        plt.plot(self.x2_list, self.y2_list, 'b--')
        plt.plot(self.x2_list[len(self.x2_list)-1], self.y2_list[len(self.y2_list)-1], 'o', c='b', marker='*')
        #plt.plot(self.x3_list, self.y3_list, 'b--')
        #plt.plot(self.x3_list[len(self.x3_list)-1], self.y3_list[len(self.y3_list)-1], 'o', c='b', marker='*')
        #plt.plot(self.x4_list, self.y4_list, 'b--')
        #plt.plot(self.x4_list[len(self.x4_list)-1], self.y4_list[len(self.y4_list)-1], 'o', c='b', marker='*')
        if(len(self.robot_movement_x_1)<8):
            plt.plot(self.robot_movement_x_1, self.robot_movement_y_1, 'r--')
        else:
            plt.plot(self.robot_movement_x_1[-8:], self.robot_movement_y_1[-8:], 'r--')
        plt.scatter(self.robot_movement_x_1[len(self.robot_movement_x_1) - 1], self.robot_movement_y_1[len(self.robot_movement_y_1) - 1], color='r', marker='D')
        if(len(self.robot_movement_x_2)<8):
            plt.plot(self.robot_movement_x_2, self.robot_movement_y_2, 'r--')
        else:
            plt.plot(self.robot_movement_x_2[-8:], self.robot_movement_y_2[-8:], 'r--')
        plt.scatter(self.robot_movement_x_2[len(self.robot_movement_x_2) - 1], self.robot_movement_y_2[len(self.robot_movement_y_2) - 1], color='r', marker='D')
        #if(len(self.robot_movement_x_3) < 8):
        #    plt.plot(self.robot_movement_x_3, self.robot_movement_y_3, 'r--')
        #else:
        #    plt.plot(self.robot_movement_x_3[-8:], self.robot_movement_y_3[-8:], 'r--')
        #plt.scatter(self.robot_movement_x_3[len(self.robot_movement_x_3) - 1], self.robot_movement_y_3[len(self.robot_movement_y_3) - 1], color='r', marker='D')
        #if(len(self.robot_movement_x_4) < 8):
        #    plt.plot(self.robot_movement_x_4, self.robot_movement_y_4, 'r--')
        #else:
        #    plt.plot(self.robot_movement_x_4[-8:], self.robot_movement_y_4[-8:], 'r--')
        #plt.scatter(self.robot_movement_x_4[len(self.robot_movement_x_4) - 1], self.robot_movement_y_4[len(self.robot_movement_y_4) - 1], color='r', marker='D')
        plt.savefig("/home/arpitdec5/Desktop/robot_target_tracking/s2/" + str(self.time_step) + ".png")
        #plt.show()


    # reference: https://matplotlib.org/3.1.0/gallery/statistics/confidence_ellipse.html
    def get_correlated_dataset(self, n, dependency, mu, scale):
        latent = np.random.randn(n, 2)
        dependent = latent.dot(dependency)
        scaled = dependent * scale
        scaled_with_offset = scaled + mu
        return (scaled_with_offset[:, 0], scaled_with_offset[:, 1])


    def compute_reward(self):
        """ 
            Function for computing the reward
        """
        val = 0.0
        for index in range(0, self.num_targets):
            val += np.log(np.linalg.det(self.estimated_targets_var[index]))
        return -val, False  


    def _reset_sim(self, **kwargs):
        raise NotImplementedError()


    def _get_obs(self):
        """ 
            Get observation function
            Returns the noisy relative measurement depending on the measurement model (bearing or range)
        """
        targets_pos = self.true_targets_pos
        sensor_pos = self.sensors_pos

        if self.meas_model=='bearing':
            true_measurement = torch.zeros(self.num_sensors, self.num_targets)
            for index in range(0, self.num_sensors):
                true_measurement[index] = torch.atan2(targets_pos[:, 1]-sensor_pos[index, 1], targets_pos[:, 0]-sensor_pos[index, 0])
        elif self.meas_model=='range':
            true_measurement = torch.zeros(self.num_sensors, self.num_targets)
            for index in range(0, self.num_sensors):
                true_measurement[index] = torch.norm(targets_pos-sensor_pos[index], p=2, dim=1)

        true_obs = true_measurement+(self.sigma_meas*torch.randn(sensor_pos.shape[0], targets_pos.shape[0]))
        return true_obs


    def get_estimated_obs(self):
        """ 
            Get observation function
            Returns the noisy relative measurement depending on the measurement model (bearing or range)
        """
        targets_pos = self.estimated_targets_mean
        sensor_pos = self.sensors_pos

        if self.meas_model == 'bearing':
            true_measurement = torch.zeros(self.num_sensors, self.num_targets)
            for index in range(0, self.num_sensors):
                true_measurement[index] = torch.atan2(targets_pos[:, 1]-sensor_pos[index, 1], targets_pos[:, 0]-sensor_pos[index, 0])
        elif self.meas_model == 'range':
            for index in range(0, self.num_sensors):
                true_measurement[index] = torch.norm(targets_pos-sensor_pos[index], p=2, dim=1)
        return true_measurement


    def update_true_targets_pos(self):
        """
            Function to update the true target positions when time step increases (assuming circular target motions)
        """
        for index in range(0, self.num_targets):
            self.true_targets_pos[index] = torch.tensor([self.true_targets_radii[index]*np.cos((self.time_step-1)/float(self.target_motion_omegas[index]))+float(self.initial_true_targets_pos[index, 0])-self.true_targets_radii[index], self.true_targets_radii[index]*np.sin((self.time_step-1)/float(self.target_motion_omegas[index]))+float(self.initial_true_targets_pos[index, 1])]) 


    def update_estimated_targets_pos(self):
        """
            Function to update the estimated target positions when time step increases (using ekf)
        """
        true_obs = self._get_obs()
        for index in range(0, self.num_targets):
            z_true = np.zeros((self.num_sensors, 1))
            for sensor_index in range(0, self.num_sensors):
                z_true[sensor_index, 0] = true_obs[sensor_index, index]         
            q_matrix = 0.2*np.eye(2)
            x_matrix = np.array([[float(self.estimated_targets_mean[index, 0])], [float(self.estimated_targets_mean[index, 1])]])
            sigma_matrix = self.estimated_targets_var[index].numpy()+q_matrix

            z_pred = np.zeros((self.num_sensors, 1))
            h_matrix = np.zeros((self.num_sensors, 2))
            for sensor_index in range(0, self.num_sensors):
                z_pred[sensor_index, 0] = np.linalg.norm([[x_matrix[0, 0]-float(self.sensors_pos[sensor_index, 0])], [x_matrix[1, 0]-float(self.sensors_pos[sensor_index, 1])]], 2)
                h_matrix[sensor_index, 0] = (-1.0/z_pred[sensor_index])*(float(self.sensors_pos[sensor_index, 0])-x_matrix[0, 0])
                h_matrix[sensor_index, 1] = (-1.0/z_pred[sensor_index])*(float(self.sensors_pos[sensor_index, 1])-x_matrix[1, 0])
            
            res = (z_true-z_pred)
            r_matrix = 1.0*1.0*np.eye(self.num_sensors)
            s_matrix = np.matmul(np.matmul(h_matrix, sigma_matrix), h_matrix.T)+r_matrix
            k_matrix = np.matmul(np.matmul(sigma_matrix, h_matrix.T), np.linalg.inv(s_matrix))

            x_matrix_tplus1 = x_matrix+(np.matmul(k_matrix, res))
            sigma_matrix_tplus1 = np.matmul(np.matmul((np.eye(2)-np.matmul(k_matrix, h_matrix)), sigma_matrix), (np.eye(2)-np.matmul(k_matrix, h_matrix)).T)+np.matmul(np.matmul(k_matrix, r_matrix), k_matrix.T)
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
        return self.estimated_targets_mean

    
    def get_posterior_map(self):
        return self.posterior_map

    
    def _set_action(self, action, step_size):
        """
            Applies the given action to the sensor.
        """
        for index in range(0, self.num_sensors):
            curr_action = torch.tensor(action[index]).float()
            vector = step_size[index]*torch.tensor([torch.cos(curr_action), torch.sin(curr_action)])
            sensor_pos = self.sensors_pos[index]+vector
            if(sensor_pos[0]>=self.len_workspace):
                sensor_pos[0] = self.sensors_pos[index, 0]
            if(sensor_pos[1]>=self.len_workspace):
                sensor_pos[1] = self.sensors_pos[index, 1]
            self.sensors_pos[index] = sensor_pos   

        
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
