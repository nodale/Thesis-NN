import numpy as np
import torch
from pyulog import ULog


ulog = ULog("rl_dataset/rl_3.ulg")
topics = {(d.name, d.multi_id): d for d in ulog.data_list}

device = "cuda"
dtype = torch.float32



def tensor(x):
    return torch.as_tensor(x.copy(), device=device, dtype=dtype)

def load_topic(topic, timestamp_key, keys):
    t = tensor(topic.data[timestamp_key] / 1e6)
    x = torch.stack([tensor(topic.data[k]) for k in keys])
    return t, x

def interp(t_src, x_src, t_dst, mode="linear"):
    if mode == "const":
        idx = torch.searchsorted(t_src, t_dst, right=True) - 1
        idx = idx.clamp(0, len(t_src) - 1)
        return x_src[:, idx]

    t_src = t_src.cpu().numpy()
    t_dst = t_dst.cpu().numpy()

    y = np.stack([
        np.interp(t_dst, t_src, x.cpu().numpy())
        for x in x_src
    ])

    return torch.as_tensor(y, device=device, dtype=dtype)



t_imu, imu = load_topic(
    topics[("vehicle_imu", 0)],
    "timestamp_sample",
    [f"delta_velocity[{i}]" for i in range(3)] +
    [f"delta_angle[{i}]" for i in range(3)]
)
t_state, states = load_topic(
    topics[("estimator_states", 0)],
    "timestamp",
    [f"states[{i}]" for i in range(12)]
)
t_action, actions = load_topic(
    topics[("johnny_status", 0)],
    "timestamp",
    [f"u[{i}]" for i in range(4)]
)
t_sp, setpoints = load_topic(
    topics[("trajectory_setpoint", 0)],
    "timestamp",
    [f"position[{i}]" for i in range(3)]
)


print(topics[("estimator_states", 0)].data["states[0]"])

hz =200
dt = 1 / hz
t0 = max(t[0] for t in [t_imu, t_state, t_action, t_sp])
tf = min(t[-1] for t in [t_imu, t_state, t_action, t_sp])
t = torch.arange(t0, tf, dt, device=device)



imu        = interp(t_imu,    imu,       t, "linear")
states     = interp(t_state,  states,    t, "linear")
actions    = interp(t_action, actions * 9.81,   t, "const")
setpoints  = interp(t_sp,     setpoints, t, "const")


dataset = torch.cat([
    states.T,            # (T, 12)
    imu.T,               # (T, 6)
    actions.T,           # (T, 4)
    setpoints.T,         # (T, 3)
    t[:, None],          # (T, 1)
], dim=1)


print(dataset.shape)  # (T, 26)
torch.save(dataset, "rl_dataset/converted.pt")

print("done")
