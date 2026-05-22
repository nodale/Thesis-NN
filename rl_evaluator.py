import torch

from pyulog import ULog

ulog = ULog("rl_dataset/rl_1.ulog")

topics = {}

for d in ulog.data_list:
    topics[(d.name, d.multi_id)] = d
    #print(d.name)


#we need : timestamp, imu[:6], vicon[:12], action[:4], setpoint[:3] 
v_sensor = topics[("vehicle_imu", 0)]

v_imu_timestamp = v_sensor.data["timestamp_sample"]/1e6
v_imu = torch.empty((6, v_imu_timestamp.shape[0]), device="cuda", dtype=torch.float32)
for i in range(3):
    v_imu[i] = torch.as_tensor(v_sensor.data[f"delta_velocity[{i}]"].copy(), device="cuda", dtype=torch.float32)
    v_imu[i+3] = torch.as_tensor(v_sensor.data[f"delta_angle[{i}]"].copy(), device="cuda", dtype=torch.float32)


v_estimate = topics[("estimator_states", 0)]
v_states = torch.empty((12, v_estimate.data["timestamp"].shape[0]), device="cuda", dtype=torch.float32)
for i in range(12):
    v_states[i] = torch.as_tensor(v_estimate.data[f"states[{i}]"].copy(), device="cuda", dtype=torch.float32)


v_johnny = topics[("johnny_status", 0)]
v_actuation = torch.empty((4, v_johnny.data["timestamp"].shape[0]), device="cuda", dtype=torch.float32)
for i in range(4):
    v_actuation[i] = torch.as_tensor(v_johnny.data[f"u[{i}]"].copy(), device="cuda", dtype=torch.float32)


v_traj_sp = topics[("trajectory_setpoint", 0)]
v_setpoint = torch.empty((4, v_traj_sp.data["timestamp"].shape[0]), device="cuda", dtype=torch.float32)
for i in range(3):
    v_setpoint[i] = torch.as_tensor(v_traj_sp.data[f"position[{i}]"].copy(), device="cuda", dtype=torch.float32)


