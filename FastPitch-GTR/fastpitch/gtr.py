"""Validated GTR labels shared by dataset and eager inference."""
from pathlib import PurePath
import re


def parse_label(label):
    label = str(label)
    if len(label) != 3 or not label.isdecimal():
        raise ValueError("GTR labels must have three digits, preserving leading zero")
    g, t, r = map(int, label)
    if not (0 <= g <= 4 and 1 <= t <= 5 and 0 <= r <= 6 and ((g == 0) == (r == 0))):
        raise ValueError(f"Illegal GTR label {label}: whisper G=0 and R=0 must be coupled")
    return g, t, r


def from_path(path):
    path = PurePath(path)
    match = re.match(r"^(\d{3})(?:_|$)", path.stem)
    if match:
        return parse_label(match.group(1))
    if len(path.parent.name) == 3 and path.parent.name.isdecimal():
        return parse_label(path.parent.name)
    raise ValueError(f"No GTR label in filename prefix or parent directory: {path}")


def validate_components(g, t, r):
    # Reject fractional controls before conversion/embedding lookup, rather than
    # silently truncating them to another GTR label.
    import numbers
    import torch
    for value in (g, t, r):
        if isinstance(value, torch.Tensor):
            if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("GTR components must be integers")
        elif not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError("GTR components must be integers")
    valid = ((g >= 0) & (g <= 4) & (t >= 1) & (t <= 5) & (r >= 0) & (r <= 6)
             & ((g == 0) == (r == 0)))
    if hasattr(valid, "all"):
        valid = valid.all()
    if not bool(valid):
        raise ValueError("Illegal GTR: G=0 and R=0 must be coupled; T must be 1..5")
