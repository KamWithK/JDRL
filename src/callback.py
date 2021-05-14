import wandb

from stable_baselines3.common.callbacks import BaseCallback

class WandBCallback(BaseCallback):
    def __init__(self, verbose: int=True, frequency=1000, ignore=["train/n_updates"]):
        super().__init__(verbose=verbose)

        wandb.init(project="jelly-drift-rl", entity="lionel-polanski", name="SAC Test", dir="..", mode="online")
        self.FREQUENCY = frequency
        self.IGNORE = ignore

    def _on_step(self) -> bool:
        # Upload frequency control
        commit = self.num_timesteps % self.FREQUENCY

        # Combine all metrics and values to be logged
        log_dict = {"train/reward": self.locals["reward"][0]} if self.locals["reward"] != None else {}
        log_dict.update(self.logger.get_log_dict())
        [log_dict.pop(key) for key in self.IGNORE if key in log_dict.keys()]

        # Log dictionary
        if log_dict: wandb.log(log_dict, commit=commit)
