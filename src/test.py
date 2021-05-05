import gym
import gym_jd

from gym_jd.envs.action_wrapper import FlattenAction
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.rescale_action import RescaleAction
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO

CONTINUOUS = True
NUM_ENVS = 5

def wrap_env(env):
    env = FlattenObservation(env)
    env = FlattenAction(env, continuous=CONTINUOUS)
    return env if not CONTINUOUS else RescaleAction(env, -1, 1)

if __name__ == "__main__":
    env = make_vec_env("jd-v0", n_envs=NUM_ENVS, wrapper_class=wrap_env, env_kwargs={"jd_path": "../Game/Jelly Drift.exe", "continuous": CONTINUOUS})

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500000)
    model.save("models/ppo")
