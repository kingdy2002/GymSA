import torch
from torch.utils.tensorboard import SummaryWriter

class Writer(object) :
    def __init__(self) :
        self.writer = SummaryWriter()


    def add(self,name,value,episode) :
        writer.add_scalar(name, value, episode)