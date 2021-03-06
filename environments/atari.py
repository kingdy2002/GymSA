from gym import Wrapper, spaces
from .wrappers import *


def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    if not 'ram' in env_id :
        env = wrap_deepmind(env)

    if 'NoFrameskip' in env.spec.id :
        env = NoopResetEnv(env, noop_max=30)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return env
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env