import wandb

from stable_baselines3.common.callbacks import BaseCallback
import plotly.graph_objects as go

class WandBCallback(BaseCallback):
    def __init__(self, verbose: int=True, frequency=1000, ignore=["train/n_updates"]):
        super().__init__(verbose=verbose)

        wandb.init(project="jelly-drift-rl", entity="lionel-polanski", name="SAC Test", dir="..", mode="online")
        self.FREQUENCY = frequency
        self.IGNORE = ignore
        self.FIRST_STEP = False

    def on_first_step(self):
        nodes = self.training_env.get_attr("BOUNDARIES")[0]
        self.path_x = []
        self.path_y = []
        self.fig = go.Figure()
        self.fig.add_scatter(x=nodes[:, 0, 2], y=nodes[:, 0, 0])
        self.fig.add_scatter(x=nodes[:, 1, 2], y=nodes[:, 1, 0])
        self.fig.add_scatter(x=[], y=[], name="path")
        self.fig.add_scatter(x=[], y=[], name="nodes", mode="markers")
        self.fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
        )

    def _on_step(self) -> bool:
        if not self.FIRST_STEP:
            self.FIRST_STEP = True
            self.on_first_step()

        # Upload frequency control
        commit = self.num_timesteps % self.FREQUENCY

        # Combine all metrics and values to be logged
        log_dict = {"train/reward": self.locals["reward"][0]} if self.locals["reward"] != None else {}
        log_dict.update(self.logger.get_log_dict())
        [log_dict.pop(key) for key in self.IGNORE if key in log_dict.keys()]

        path = self.fig.data[2]
        position = self.locals["infos"][0]["position"]
        self.path_x.append(position[2])
        self.path_y.append(position[0])
        path.x = self.path_x
        path.y = self.path_y
        wandb.log({"position": self.fig})

        # Log dictionary
        if log_dict: wandb.log(log_dict, commit=commit)
