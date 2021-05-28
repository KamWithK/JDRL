import os
import re
import stable_baselines3

import numpy as np

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

def create_model(env, type, policy, load, verbose=0):
    if isinstance(load, str):
        print(f"LOADING SAVE {load}")
        return stable_baselines3.__dict__[type].load(load, env, verbose=verbose)
    else:
        print("TRAINING FROM SCRATCH")
        return stable_baselines3.__dict__[type](policy, env, verbose=verbose)

def get_latest(check: bool):
    if not check:
        return False
    elif os.path.exists("final_model.zip"):
        return "final_model.zip"
    elif not os.path.exists("checkpoints") or os.listdir("checkpoints") == []:
        return False
    else:
        file_names = os.listdir("checkpoints")
        file_numbers = [int(re.findall("\d+", file_name)[0]) for file_name in file_names]

        return "checkpoints/" + file_names[np.argmax(file_numbers)]
