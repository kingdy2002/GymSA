from agent.Base import agent_base
class DQN(agent_base) :
    
    def __init__(self,config) :
        agent_base.__init__(self,config)
    


    def epsilon(self,episode,max_episode) :
        eps = self.config.min_eps
        return eps
