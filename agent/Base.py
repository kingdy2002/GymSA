import torch

class agent_base(object) :
    def __init__(self,config) :
        self.config = config
        self.hyperparameters = config.hyperparameters

        self.action_space = config.env.action_space
        self.action_high = config.env.action_space.high
        self.action_low = config.env.action_space.low

        self.observation_space = config.env.observation_space
        self.observation_space_high = config.env.observation_space.high
        self.observation_space_low = config.env.observation_space.low

        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.now_epi = 0
        self.tot_step = 0
        self.epi_step = 0

        self.network = None

  
        self.optim = torch.optim.Adam(self.network.parameters(), \
                lr=self.hyperparameters[lr])


    def update(self,batch) :

        return loss
        pass

    def predict(self,observation) :
        pass

    def select_action(self,observation,etc) :
        pass

    def td_error(self,state, next_state) :
        pass

    def save_model(self,save_path) :
        pass

    def load_model(self,load_path) :
        pass