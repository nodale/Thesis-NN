import numpy as np
import torch
from pyulog import ULog


ulog = ULog("rl_dataset/rl_5.ulg")
topics = {(d.name, d.multi_id): d for d in ulog.data_list}

device = "cuda"
dtype = torch.float32

def clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def tensor(x):
    return torch.as_tensor(x.copy(), device=device, dtype=dtype)

def load_topic(topic, timestamp_key, keys):
    t = clean(tensor(topic.data[timestamp_key] / 1e6))
    x = torch.stack([clean(tensor(topic.data[k])) for k in keys], dim=0)
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



t_acc, acc = load_topic(
    topics[("vehicle_acceleration", 0)],
    "timestamp_sample",
    [f"xyz[{i}]" for i in range(3)]
)
t_gyro, gyro = load_topic(
    topics[("vehicle_angular_velocity", 0)],
    "timestamp_sample",
    [f"xyz[{i}]" for i in range(3)]
)
t_state, states = load_topic(
    topics[("vehicle_odometry", 0)],
    "timestamp_sample",
    [f"position[{i}]" for i in range(3)] +
    [f"velocity[{i}]" for i in range(3)] +
    [f"q[{i}]" for i in range(4)] +
    [f"angular_velocity[{i}]" for i in range(3)] 
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




hz =200.0
dt = 1.0 / hz
t0 = max(t[0] for t in [t_acc, t_gyro, t_state])
tf = min(t[-1] for t in [t_acc, t_gyro, t_state])
t = torch.arange(t0, tf, dt, device=device)



acc        = interp(t_acc,    acc,       t, "linear")
gyro       = interp(t_gyro,    gyro,     t, "linear")
states     = interp(t_state,  states,    t, "linear")
actions    = interp(t_action, actions,   t, "const")
setpoints  = interp(t_sp,     setpoints, t, "const")

states.T[:, :3] /= 3.0
states.T[:, 3:6] /= 0.8
states.T[:, 10:13] /= 0.5

dataset = torch.cat([
    states.T,            # (T, 13)
    acc.T/25.0,               # (T, 3)
    actions.T,           # (T, 4)
    setpoints.T/3.0,         # (T, 3)
], dim=1)


#t[:, None]*0.0,          # (T, 1)

#torch.save(dataset[7000:10500], "rl_dataset/converted.pt")
torch.save(dataset[7000:], "rl_dataset/converted.pt")

print(dataset.shape)  # (T, 27)
print("done")
