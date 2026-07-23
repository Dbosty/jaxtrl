import wandb
from torchrl.record import WandbLogger as _WandbLogger


class WandbLogger(_WandbLogger):
    def log_video(self, name: str, video, **kwargs):
        fps = kwargs.pop("fps", self.video_fps)
        caption = kwargs.pop("caption", None)
        format = kwargs.pop("format", "mp4")
        self.experiment.log(
            {name: wandb.Video(video, fps=fps, format=format, caption=caption)},
            **kwargs,
        )

    def log_scalars(
        self, values: dict[str, any], **kwargs
    ):  # for wandb - setting x axis to be frames instead of step
        wandb.log(values, **kwargs)


class PrintLogger:
    def log_scalars(self, values: dict[str, any], timestep: int = None, **kwargs):
        self._print_values(values, timestep)

    def log_video(self, name: str, video, timestep: int = None, **kwargs):
        print(f"[INFO] Video log attempt for {name} at timestep {timestep}")

    def _print_values(self, values, timestep):
        print("\n")
        print("-" * 100)
        print(f"Printing values for timestep {timestep}")
        print("-" * 100)

        values = self._nest_dict_from_slash_keys(values)
        self._print_nested(values)

        print("-" * 100)
        print("\n")

    def _nest_dict_from_slash_keys(self, flat_dict):
        nested = {}
        for k, v in flat_dict.items():
            keys = k.split("/")
            d = nested
            for ki in keys[:-1]:
                if ki not in d or not isinstance(d[ki], dict):
                    d[ki] = {}
                d = d[ki]
            d[keys[-1]] = v
        return nested

    def _print_nested(self, d, indent=0):
        indent_str = "\t" * indent
        if isinstance(d, dict):
            for k, v in d.items():
                print(
                    f"{indent_str}{k}/"
                    if isinstance(v, dict)
                    else f"{indent_str}{k}: {v}"
                )
                if isinstance(v, dict):
                    self._print_nested(v, indent + 1)
        else:
            print(f"{indent_str}{d}")


class ExtWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_logger = PrintLogger()

    def log_scalars(self, values, timestep: int = None, **kwargs):
        super().log_scalars(values, **kwargs)
        self.print_logger.log_scalars(values, timestep)

    def log_video(self, name: str, video, timestep: int = None, **kwargs):
        super().log_video(name, video, **kwargs)
        self.print_logger.log_video(name, video, timestep)
