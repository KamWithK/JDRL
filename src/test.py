import gym_jd
import stable_baselines3

from gym_jd.envs.action_wrapper import FlattenAction
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.rescale_action import RescaleAction
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from callback import WandBCallback

CONTINUOUS = True
MODEL_TYPE = "SAC"
NUM_ENVS = 3 if MODEL_TYPE != "SAC" else 1
# LOAD, MAX_TIMESTEPS = False, 1000000
LOAD, SAVE = False, False
MAX_TIMESTEPS = 1000000 if SAVE else 10000

SAVE_PATHS = {
    "training": "models/training", "training_prefix": MODEL_TYPE.lower(),
    "final": f"models/final/{MODEL_TYPE.lower()}"
}

def wrap_env(env):
    env = FlattenObservation(env)
    env = FlattenAction(env, continuous=CONTINUOUS)
    return env if not CONTINUOUS else RescaleAction(env, -1, 1)

if __name__ == "__main__":
    env = make_vec_env("jd-v0", n_envs=NUM_ENVS, wrapper_class=wrap_env, env_kwargs={"jd_path": "../Game/Jelly Drift.exe", "continuous": CONTINUOUS})

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=SAVE_PATHS["training"], name_prefix=SAVE_PATHS["training_prefix"]) if SAVE else None
    wandb_callback = WandBCallback()
    callbacks = list(filter(None, [wandb_callback, checkpoint_callback]))

    model = stable_baselines3.__dict__[MODEL_TYPE].load(SAVE_PATHS["final"], env, verbose=1) if LOAD else stable_baselines3.__dict__[MODEL_TYPE]("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=MAX_TIMESTEPS, callback=callbacks)
    if SAVE: model.save(SAVE_PATHS["final"])
