import torch.nn as nn
import torch
import torch.nn.functional as F
import modules

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, state_dim)
        self.fc2 = nn.Linear(state_dim,action_dim *4)
        self.fc3 = nn.Linear(action_dim *4, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

class Actor_img(nn.Module):
    def __init__(self, action_dim):
        super(Actor_img, self).__init__()
        self.cnn = modules.cnn.Net()
        self.fc1 = nn.Linear(1600, 800)
        self.fc2 = nn.Linear(800,128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_value = self.fc3(x)
        return action_value

class Net(nn.Module):
    def __init__(self, observation_space, action_space,env_name):
        super(Net, self).__init__()
        self.action_dim = action_space.n
        if 'ram' in env_name :
            self.obs_dim = observation_space.shape[0]
            self.net = Actor(self.obs_dim, self.action_dim)
        else :
            self.obs_dim = observation_space.shape[-1]
            self.net = Actor_img(self.action_dim)


    def forward(self, x):
        x = self.net(x)
        return x