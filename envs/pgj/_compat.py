"""Runtime environment requirements for the Algorithm 1/2/3 scripts in this
directory (level_k.py, pap.py, lksp.py).

These are not things `pip install` can pin for you -- they're env vars that
change the *behavior* of already-installed packages. Get them wrong and
nothing fails at import time; training just breaks partway through with a
confusing error (or, worse, silently computes the wrong thing).

COMPOSITE_LP_AGGREGATE=0
    jaxtrl's HAPPO/MAPPO/PPO code (jaxtrl/src/jaxtrl/marl/*.py) writes the
    PPO log-prob into td["action_log_prob"] -- the torchrl convention from
    when jaxtrl was written. Current tensordict/torchrl default
    `composite_lp_aggregate()` to True, under which GAE/ClipPPOLoss instead
    look for td["sample_log_prob"], so training dies with
    `KeyError: "Couldn't find the log-prob sample_log_prob in the input
    data."`. Setting this env var to 0 restores the old default for this
    process. Must be set in the shell *before* the interpreter starts (it's
    read by tensordict/torchrl internals, not by this file) -- exporting it
    inside Python after `import torch`/`import torchrl` is too late.

    Set it once for the conda env so you don't have to remember it per
    command:
        conda env config vars set COMPOSITE_LP_AGGREGATE=0 -n peg
        conda deactivate && conda activate peg

Known-good, mutually-compatible package versions (verified 2026-09-15;
neither jaxtrl's nor pgj's pyproject.toml pins these, so a fresh `pip
install` can and will drift to incompatible combinations):
    torch==2.7.0+cu126
    torchrl==0.8.1        (first release whose GAE supports
                           `deactivate_vmap`, which jaxtrl's MAPPO/HAPPO
                           construction requires)
    tensordict==0.8.3     (version range torchrl==0.8.1 declares)
    jax[cuda12]==0.10.2
    nvidia-cudnn-cu12>=9.8,<10.0
                          (jax's floor; torch declares an older exact pin,
                           but cuDNN 9.x keeps ABI compatibility across
                           minors -- reinstall this cudnn version *last*,
                           since reinstalling torch/torchrl drags an older
                           one back in as a transitive dependency)

If you rebuild this environment from scratch, install torch+torchrl+
tensordict together in one pinned `pip install` command (so pip can't
silently resolve a newer, incompatible torch on its own), then bump
nvidia-cudnn-cu12 afterward.
"""

import os
import sys

_REQUIRED = {"COMPOSITE_LP_AGGREGATE": "0"}


def check_env():
    missing = {k: v for k, v in _REQUIRED.items() if os.environ.get(k) != v}
    if missing:
        pairs = ", ".join(f"{k}={v}" for k, v in missing.items())
        sys.exit(
            f"Missing/incorrect required environment variable(s): {pairs}\n"
            f"Set them before running this script (see envs/pgj/_compat.py "
            f"for why), e.g.:\n"
            f"    export {' '.join(f'{k}={v}' for k, v in missing.items())}\n"
            f"or persist them on the conda env:\n"
            f"    conda env config vars set {' '.join(f'{k}={v}' for k, v in missing.items())}"
        )


check_env()
