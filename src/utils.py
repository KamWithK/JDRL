import stable_baselines3

from gym.wrappers.flatten_observation import FlattenObservation
from gym_jd.utils.action_wrapper import FlattenAction
from gym.wrappers.rescale_action import RescaleAction

def wrap_continuous_env(env):
    env = FlattenObservation(env)
    env = FlattenAction(env, continuous=True)
    env = RescaleAction(env, -1, 1)

    return env

def wrap_discrete_env(env):
    env = FlattenObservation(env)
    env = FlattenAction(env, continuous=False)

    return env

def create_model(env, type, policy, load, save_path="final_model", verbose=0):
    if load:
        return stable_baselines3.__dict__[type].load(save_path, env, verbose=verbose)
    else:
        return stable_baselines3.__dict__[type](policy, env, verbose=verbose)