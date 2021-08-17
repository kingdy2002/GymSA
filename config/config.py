class Config(object):
    def __init__(self):
        self.seed = None

        self.env = None
        self.env_name = None
        self.env_observation = 'image' #image or vector
        self.env_args = {}

        self.agent_name = None

        self.epsilon = None
        self.max_eps = 1
        self.min_eps = 0.05

        self.update_interval = 1 #interval of update network at step
        self.logging_interval = 20 #interval of logging at epi
        self.max_epi = None
        self.hyperparameters = {}

        self.test_epi = None
        self.test_interval = None #interval of test at epi

        self.save_model = True
        self.load_path = ''
        self.save_path = ''

        self.evaluate = False

        self.replay = True #use replay buffer?
        self.replay_buffer = 'replay_buffer'  #replay_buffer or per
