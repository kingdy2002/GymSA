from agent.Base import agent_base
from modules.value_base import dqn_model
import torch
import random
import numpy as np
import torch.functional as F


class dqn(agent_base) :
    
    def __init__(self,config) :
        agent_base.__init__(self,config)
        self.network = dqn_model.Net(self.observation_space, self.action_space, config.env_name).to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), \
                lr=self.hyperparameters['lr'])

    def epsilon(self,episode,max_episode) :
        eps_ = self.config.min_eps + (self.config.max_eps - self.config.min_eps)*episode/max_episode*0.75
        eps = np.min(self.config.max_eps, eps_)
        return eps


    def predict(self, observation):
        if not isinstance(observation, list):
            observation = [observation]
        observation = torch.tensor(observation).to(self.device)
        with torch.no_grad :
            self.network.eval()
            Q = self.network(observation)
            self.network.train()
        return Q

    def select_action(self,observation,episode):
        eps = self.epsilon(episode , self.config.max_episode)
        random_ac = random.randint(0,self.action_space - 1)
        
        if random.random() > eps :
            ac = random_ac
        else :
            Q_ = self.predict(observation)
            ac = torch.argmax(Q_,dim = 1).item().detach()
        return ac


    def td_error(self,states, actions, rewards,next_states, dones) :
        if not isinstance(states, list):
            states = [states]
        if not isinstance(actions, list):
            actions = [actions]
        if not isinstance(rewards, list):
            rewards = [rewards]
        if not isinstance(next_states, list):
            next_states = [next_states]
        if not isinstance(dones, list):
            dones = [dones]

        target_Q = self.network(next_states).detach().max(1)[0].unsqueeze(1)
        target_Q = rewards + (self.hyperparameters["discount_rate"] * target_Q * (1 - dones))

        policy_Q = self.network(states).gather(1, actions.long())
        td_error = F.mse_loss(target_Q, policy_Q)

        return td_error

    def update(self, batch):

        states_batch = torch.from_numpy(np.vstack([b.state for b in batch if b is not None])).float().to(self.device)
        actions_batch = torch.from_numpy(np.vstack([b.action for b in batch if b is not None])).float().to(self.device)
        rewards_batch = torch.from_numpy(np.vstack([b.reward for b in batch if b is not None])).float().to(self.device)
        next_states_batch = torch.from_numpy(np.vstack([b.next_state for b in batch if b is not None])).float().to(self.device)
        dones_batch = torch.from_numpy(np.vstack([int(b.done) for b in batch if b is not None])).float().to(self.device)

        loss = self.td_error(states_batch,actions_batch,rewards_batch,next_states_batch,dones_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

