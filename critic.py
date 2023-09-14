import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from common import MLP


class ValueCritic2(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_layers,
        **kwargs
    ) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, 1, hidden_dim, n_layers, **kwargs)

    def forward(self, state):
        return self.mlp(state)


class Critic2(nn.Module):
    """
    From TD3+BC
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

#NEWLY ADDED

class ValueCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #cov2d layers NEW
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding =0) # out size 38x38x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,stride=1, padding =0) #36x36x8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #34x34x4
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,stride=2, padding =1) #16x16x4
        self.flatten= nn.Flatten()
        self.fc1 = nn.Linear(900, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.LayerNorm(int(hidden_size/2))
        self.mu = nn.Linear(int(hidden_size/2), 1)
        
        # self.fc1 = nn.Linear(state_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)

        # self.mu = nn.Linear(hidden_size, action_size)
        # self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # print("State in act net:",state.shape)
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=self.flatten(x)
        # print('flatten shape :',x.shape)
        x=self.fc1(x)
        x=F.relu(self.bn1(x))
        x=self.fc2(x)
        x=F.relu(self.bn2(x))
        mu = self.mu(x)

        return mu

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, 1)
        # self.reset_parameters()

        # Q1 architecture
        #cov2d layers NEW
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding =0) # out size 38x38x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,stride=1, padding =0) #36x36x8
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #34x34x4
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,stride=2, padding =1) #16x16x4
        self.flatten= nn.Flatten()
        self.fc1 = nn.Linear(900, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn2 = nn.LayerNorm(int(hidden_size/2))

        self.fc3 = nn.Linear(action_size,int(hidden_size/4))
        self.bn3 = nn.LayerNorm(int(hidden_size/4))

        self.fc4 = nn.Linear(int(hidden_size/2)+int(hidden_size/4),int(hidden_size/4))
        self.bn4 = nn.LayerNorm(int(hidden_size/4))

        self.fc_out = nn.Linear(int(hidden_size/4), 1)


        # Q2 architecture
        #cov2d layers NEW
        self.conv12 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding =0) # out size 38x38x8
        self.conv22 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5,stride=1, padding =0) #36x36x8
        self.conv32 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5,stride=1, padding =1) #34x34x4
        self.conv42 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5,stride=2, padding =1) #16x16x4
        self.flatten2= nn.Flatten()
        self.fc12 = nn.Linear(900, hidden_size)
        self.bn12 = nn.LayerNorm(hidden_size)
        self.fc22 = nn.Linear(hidden_size, int(hidden_size/2))
        self.bn22 = nn.LayerNorm(int(hidden_size/2))

        self.fc32 = nn.Linear(action_size,int(hidden_size/4))
        self.bn32 = nn.LayerNorm(int(hidden_size/4))

        self.fc42 = nn.Linear(int(hidden_size/2)+int(hidden_size/4),int(hidden_size/4))
        self.bn42 = nn.LayerNorm(int(hidden_size/4))

        self.fc_out2 = nn.Linear(int(hidden_size/4), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # x = torch.cat((state, action), dim=-1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        #state
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=self.flatten(x)
        x=self.fc1(x)
        x=F.relu(self.bn1(x))
        x=self.fc2(x)
        x=F.relu(self.bn2(x))

        #action
        y=self.fc3(action)
        y=F.relu(self.bn3(y))

        z = torch.cat((x, y), dim=-1)
        # print('concat shape :',z.shape)
        z=self.fc4(z)
        z=F.relu(self.bn4(z))
        q1 = self.fc_out(z)


        #state
        x2 = F.relu(self.conv12(state))
        x2 = F.relu(self.conv22(x2))
        x2 = F.relu(self.conv32(x2))
        x2 = F.relu(self.conv42(x2))
        x2=self.flatten(x2)
        x2=self.fc12(x2)
        x2=F.relu(self.bn12(x2))
        x2=self.fc22(x2)
        x2=F.relu(self.bn22(x2))

        #action
        y2=self.fc32(action)
        y2=F.relu(self.bn32(y2))

        z2 = torch.cat((x2, y2), dim=-1)
        # print('concat shape :',z.shape)
        z2=self.fc42(z2)
        z2=F.relu(self.bn42(z2))
        q2 = self.fc_out2(z2)

        return q1, q2