import gym
import environments
import agent
import config.config
import utills
import modules
import torch


class episode_run(object) :

    def __init__(self,config,logger , agent) :
        self.config = config
        self.logger = logger
        self.agent = agent
        self.writer = utills.result.Writer
        self.env = config.env
        self.env_name = config.env_name

        self.logger.print_info('let start %s' % self.env_name)

        self.hyperparameters = config.hyperparameters
        
        self.batch_size = self.hyperparameters[batch_size]
        
        if config.replay :
            if config.replay_buffer == 'replay_buffer' :
                self.memory = modules.memory.replay_buffer

            if config.replay_buffer == 'per' :
                self.memory = modules.memory.per


        self.step = 0
        self.episode = 0
        self.Timer = utills.timer()

        self.score = []
        self.episode_lis = []

        self.run()

    def run(self) :

        done = False
        if config.evaluate :
            self.test(self.config.test_epi)
            return
        

        for self.episode in range(self.config.max_epi) :
            state = self.env.reset()
            self.step = 0
            total_reward = 0
            self.logger.reset_stat()
            self.logger.log_stat('episode',self.episode, 0)
            self.Timer.start_episode()
            train_time = 0
            while not done :
                self.step = self.step + 1
                self.env.render()

                if self.config.epsilon :
                    epsilon = self.agent.epsilon(self.episode, self.config.max_epi)
                    action = self.agent.select_action(state,epsilon)
                else :
                    action = self.agent.select_action(state)
                next_state, reward, done, info = self.agent.step(action)

                if config.replay :
                    if config.replay_buffer == 'replay_buffer' :
                        self.memory.push(state, next_state, action, reward, done)

                    if config.replay_buffer == 'per' :
                        td_error = self.agent.td_error(state, next_state)
                        self.memory.push(td_error,state, next_state, action, reward, done)
                
                state = next_state
                total_reward = total_reward + reward

                if self.step % self.config.update_interval == 0 :

                    if config.replay :
                        if len(self.memory) > self.batch_size :
                            batch = self.memory.make_batch(self.batch_size)
                            loss = self.agetn.update(batch)
                            self.logger.log_stat('loss',loss,self.step)
                    else :
                        loss = self.agetn.update(state, next_state, action, reward, done)
                        self.logger.log_stat('loss',loss,self.step)

            episode_t = self.Timer.finish_episode()
            train_time = train_time + self.Timer.cal_time(episode_t)

            self.writer.add('score',total_reward,self.episode)

            self.logger.log_stat('episode_len',self.step,0)
            self.logger.log_stat('total_reward',total_reward,0)

            if self.config.test_interval != None or self.episode % self.config.test_interval == 0 :
                self.test(self.config.test_epi)

            if self.episode % self.config.logging_interval == 0 :
                
                hour,minute,sec = self.Timer.cal_time(episode_t)
                episode_time = 'episode {} : operate {} hours, {} minute, {} seconde'.format(self.episode,hour,minute,sec)
                self.logger.print_info(episode_time)

                left_epi = self.config.max_ep - self.episode
                left_time = left_epi * train_time / self.config.logging_interval
                hour,minute,sec = self.Timer.cal_time(left_time)
                left_time_ = 'left train time : {} hours, {} minute, {} seconde'.format(self.episode,hour,minute,sec)

                self.logger.print_info(left_time_)

                self.logger.print_recent_stats()


        t = self.Timer.finish_train()
        hour,minute,sec = self.Timer.cal_time(t)
        train_time = 'total operate time : {} hours, {} minute, {} seconde'.format(self.episode,hour,minute,sec)
        self.logger.print_info(train_time)

        self.episode_lis.append(self.episode)
        self.score.append(total_reward)

    def test(self,test_epi) :
        pass