"""Shared helpers so every figure-producing script (1) keeps a plain-text
record of the numbers behind its plot, alongside the .png, and (2) uses the
same color for a given algorithm (PAP/MPO/CBO-DAgger/CBO-Offline) in every
figure across the repo."""

ALGO_COLORS = {
    "PAP": "tab:blue",
    "MPO": "tab:orange",
    "CBO-DAgger": "tab:green",
    "CBO-Offline": "tab:red",
    "Random": "tab:gray",
}


def save_values_txt(path, **series):
    """series: name -> scalar, or name -> (list/array of x, list/array of y)
    for a 2-column series, or name -> dict for a name->value mapping. Written
    as simple, greppable "key: value" lines -- not meant to be re-parsed
    programmatically (evaluate.py/evaluate_acc.py already .npy-save anything
    that needs that), just a human-readable audit trail of what went into
    the figure."""
    with open(path, "w") as f:
        for name, value in series.items():
            f.write(f"# {name}\n")
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"{k}\t{v}\n")
            elif isinstance(value, tuple) and len(value) == 2:
                xs, ys = value
                for x, y in zip(xs, ys):
                    f.write(f"{x}\t{y}\n")
            elif isinstance(value, tuple) and len(value) == 3:
                xs, ys, errs = value
                for x, y, e in zip(xs, ys, errs):
                    f.write(f"{x}\t{y}\t{e}\n")
            elif hasattr(value, "__iter__") and not isinstance(value, str):
                for v in value:
                    f.write(f"{v}\n")
            else:
                f.write(f"{value}\n")
            f.write("\n")
