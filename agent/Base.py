import torch

class agent_base(object) :
    def __init__(self,config) :
        self.config = config
        self.hyperparameters = config.hyperparameters

        self.action_space = config.env.action_space

        self.observation_space = config.env.observation_space
        self.observation_space_high = config.env.observation_space.high
        self.observation_space_low = config.env.observation_space.low

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.now_epi = 0
        self.tot_step = 0
        self.epi_step = 0

        self.network = None




    def update(self,batch) :

        #return loss
        pass

    def predict(self,observation) :
        # return pure result of network
        pass

    def select_action(self,observation) :
        pass

    def td_error(self,states, next_states, rewards, actions, dones) :
        pass

    def save_model(self,save_path) :
        pass

    def load_model(self,load_path) :
        pass