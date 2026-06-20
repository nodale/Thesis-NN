import torch

def denormalise(x, cfg=None):
    """
    x: torch.Tensor [..., C]
    Assumes fixed channel layout.
    """
    x = x.clone()

    # --- position ---
    x[..., 0:3] *= 3.0

    # --- linear velocity ---
    x[..., 3:6] *= 0.8

    # --- quaternion (FRD) ---
    # no scaling
    # x[..., 6:10] unchanged (if present)

    # --- angular velocity ---
    x[..., 10:13] *= 0.5

    # --- acceleration ---
    x[..., 13:16] *= 25.0

    # --- thrust ---
    x[..., 16:20] *= 9.81

    # --- setpoint ---
    x[..., 20:23] *= 3.0

    return x
