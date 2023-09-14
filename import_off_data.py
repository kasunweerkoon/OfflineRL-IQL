from torchvision import datasets, transforms
import os
import numpy as np
import math
from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImportData(Dataset):
	def __init__(self,path):
		self.odom_goal =[]
		self.odom_current = []
		self.tot_dist_to_goal = 0
		self.init_position_no =0
		self.state_no=0
		self.goal_reach_threshold = 0.1

		self.path = path #'/media/kasun/Media/offRL/dataset/' #'/home/kasun/offlineRL/dataset/'
		self.folder_name = 'ap25/'
		self.file_name = 'umd_ap25_'
		self.bag_no = str(1)
		self.eps_folder_name = 'episodes_v2/'

		self.dataset = []

		self.load_data()


	def load_data(self):

		lst = os.listdir(self.path+self.eps_folder_name) # list of samples in your directory path
		lst.sort()
		num_samples= len(lst)

		# num_samples = 4427 #bag 10 for now

		##state = [intensity_map, height_map, odom, joints, goal_map]
		##sample = [current_state, action, reward, next_state, done]

		# for fname in range(num_samples):
		for fname in lst:	
			# sample = np.load(self.path+self.eps_folder_name+'ep'+str(self.ep_no)+'_sample'+str(num_samples)+'.npy')
			# print(fname)		
			sample = np.load(self.path+self.eps_folder_name+fname,allow_pickle = True)		
			self.dataset.append([sample[0], sample[1], sample[2], sample[3], sample[4]])

		len(self.dataset)
		

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		current_state,action, reward, next_state,dones = self.dataset[idx]
		transform = transforms.Compose([transforms.Resize(280),
		                    transforms.CenterCrop(280),
		                    transforms.ToTensor()])

		tramsform2 = transforms.ToTensor()

		# # img_original = tramsform2(img.copy())
		# img = transform(img)		
		# odom = torch.from_numpy(odom)
		# joint = torch.from_numpy(joint)
		# action = torch.from_numpy(action)

		# print("state shape:",current_state.shape)
		# print("actionshape:",action.shape)
		# print("reward shape:",reward.shape)
		# print("dones shape:",dones.shape)		

		current_state = torch.squeeze(torch.from_numpy(current_state).T)
		next_state = torch.squeeze(torch.from_numpy(next_state).T)
		action = torch.squeeze(torch.from_numpy(action).T)
		reward = torch.unsqueeze(torch.from_numpy(reward),0)
		dones = torch.unsqueeze(torch.from_numpy(dones),0)

		# print("state shape:",current_state.shape)
		# print("actionshape:",action.shape)
		# print("reward shape:",reward.shape)
		# print("dones shape:",dones.shape)


		return (current_state, action, reward, next_state, dones) #(img,odom,joint,action)


if __name__ == '__main__':

    dataset = ImportData('/media/kasun/Media/offRL/dataset/')

