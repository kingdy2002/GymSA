import runner
from agent import policy_base
from agent import value_base
from config.config import Config
from environments.atari import make_atari
import torch
import utills.logging
import gym


log_ = utills.logging.get_logger()
log_ = utills.logging.Logger(log_)
config_ = Config()

env_name = 'SpaceInvaders-v0'
#'Breakout-v0'

env = make_atari(env_name,max_episode_steps=10000)
#env= gym.make('LunarLander-v2')

seed = 7
torch.manual_seed(seed)
env.seed(seed)

config_.env_observation = 'image'
config_.env = env
config_.env_name =  env_name
config_.env_args['max_episode_steps'] = 10000
config_.env_args['action_space'] = env.action_space
config_.env_args['observation_space'] = env.observation_space

config_.max_epi = 10000
config_.save_path = 'D:/GymSA/result'

config_.agent_name = 'dqn'
config_.epsilon = True
config_.hyperparameters['batch_size'] = 128
config_.hyperparameters['buffer_size'] = 100000
config_.hyperparameters['lr'] = 1.5*10e-4
config_.hyperparameters['discount_rate'] = 0.99

log_.print_info('play environment is {} and agent is {}'.format(env_name, config_.agent_name))
log_.print_info('action space is {}'.format(env.action_space))
log_.print_info('observation_space is {}'.format(env.observation_space))

agent_ = value_base.dqn.dqn(config_) # agent 종류에 맞추어 설정
runner_ = runner.episode_run_atari.episode_run(config_,log_,agent_)
runner_.run()