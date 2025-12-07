import wandb
from torchrl.record import WandbLogger as _WandbLogger
from torchrl.record.loggers.common import Logger


class WandbLogger(_WandbLogger):
    def log_video(self, name: str, video, **kwargs):
        fps = kwargs.pop("fps", self.video_fps)
        caption = kwargs.pop("caption", None)
        format = kwargs.pop("format", "mp4")
        self.experiment.log(
            {name: wandb.Video(video, fps=fps, format=format)},
            **kwargs,
        )
        self.experiment.log(
            {name: wandb.Video(video, fps=fps, format=format, caption=caption)},
            # step=step,
            **kwargs,
        )

    def log_scalars(
        self, values: dict[str, any], **kwargs
    ):  # for wandb - setting x axis to be frames instead of step
        wandb.log(values, **kwargs)


class PrintLogger(Logger):
    def log_scalars(self, values: dict[str, any], **kwargs):
        pass
