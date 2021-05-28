import hydra

from utils import wrap_continuous_env, wrap_discrete_env, get_latest
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call, get_original_cwd

OmegaConf.register_new_resolver("parse_string", lambda input : input.lower().replace(" ", "_"))
OmegaConf.register_new_resolver("get_wrapper_func", lambda continuous : wrap_continuous_env if continuous else wrap_discrete_env)
OmegaConf.register_new_resolver("original_dir", lambda relative_path : get_original_cwd() + relative_path)
OmegaConf.register_new_resolver("get_latest", get_latest)
OmegaConf.register_new_resolver("wandb_mode", lambda save : "online" if save else "disabled")

@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    env = call(config.environment)
    callbacks = list(instantiate(config.callbacks)["callbacks"])
    model = call(config.model, env)

    model.learn(total_timesteps=config.run["max_timesteps"], callback=callbacks)
    if config.run["save"]: model.save(config.model["save_path"])

if __name__ == "__main__":
    main()
