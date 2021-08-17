from collections import namedtuple, deque
import torch
import random

class replay_buffer(object):

    def __init__(self,batch_size,buffer_size) :
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("experience", field_names=["state", "action",\
                                                                "reward", "next_state", "done"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self,states, actions, rewards, next_states, dones) :
        if type(dones) == list:
            assert type(dones[0]) != list
            exp = [self.experience(state,action,rewad,next_state) for  state,action,rewad,next_state in \
                   zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(exp)

        else :
            exp = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(exp)

    def make_batch(self) :
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            return batch

        else :
            batch = random.sample(self.memory, len(self.memory))
            return batch



    def __len__(self) :
        return len(self.memory)

class per (object) :
    pass
