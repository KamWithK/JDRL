import wandb

from stable_baselines3.common.callbacks import BaseCallback
import plotly.graph_objects as go
from wandb.trigger import reset

class WandBCallback(BaseCallback):
    def __init__(self, verbose: int=True, frequency=1000, ignore=["train/n_updates"], mode="online", name="", resume=False):
        super().__init__(verbose=verbose)
        
        self.FREQUENCY = frequency
        self.IGNORE = ignore

        wandb.init(project="jelly-drift-rl", entity="lionel-polanski", name=name, dir="..", mode=mode, resume=resume)

    def _on_training_start(self) -> None:
        boundaries = self.training_env.get_attr("NODES")[0].BOUNDARIES

        self.fig = go.Figure()
        self.fig.add_scatter(x=boundaries[:, 0, 2], y=boundaries[:, 0, 0])
        self.fig.add_scatter(x=boundaries[:, 1, 2], y=boundaries[:, 1, 0])
        self.fig.add_scatter(x=[], y=[], name="path", mode="markers")
        self.fig.add_scatter(x=[], y=[], name="nodes", mode="markers")
        self.fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )

    def _on_step(self) -> bool:
        # Upload frequency control
        commit = self.num_timesteps % self.FREQUENCY

        # Combine all metrics and values to be logged
        log_dict = {"train/reward": self.locals["reward"][0]} if self.locals["reward"] != None else {}
        log_dict.update(self.logger.get_log_dict())
        [log_dict.pop(key) for key in self.IGNORE if key in log_dict.keys()]

        if self.locals["done"][0]:
            path = self.fig.data[2]
            position = self.locals["infos"][0]["position"]
            path.x, path.y = path.x + (position[2],), path.y + (position[0],)
            log_dict.update({"position": self.fig})

        # Log dictionary
        if log_dict: wandb.log(log_dict, commit=commit)
