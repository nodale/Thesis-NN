import sys
sys.path.insert(0, "/home/egghead/Thesis/Thesis-NN")

import matplotlib
import torch
import zarr
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

from QuickDataset import QuickDatasetStraight
from include.central_denormaliser import denormalise

def load_trajectory(path, episode_idx, pred_dim=6, window_size=None):
    if window_size is None:

        root = zarr.open(path, mode="r")
        data = root["episodes"]
        window_size = data.shape[0]

    dataset = QuickDatasetStraight(
        path=path,
        episode_idx=episode_idx,
        window_size=window_size,
    )

    positions = []

    for sample in dataset:
        positions.append(sample[0, :pred_dim])

    positions = torch.stack(positions)
    positions = denormalise(positions)

    return positions.cpu().numpy()

def plot_trajectories(trajectories, pred_dim=3, window_size=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for traj_info in trajectories:
        path = traj_info["path"]
        episode_idx = traj_info["episode"]
        label = traj_info.get(
            "label",
            f"{path.split('/')[-1]} - Ep {episode_idx}"
        )

        traj = load_trajectory(
            path=path,
            episode_idx=episode_idx,
            pred_dim=pred_dim,
            window_size=window_size,
        )
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            label=label,
        )
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            traj[0, 2],
            marker="o",
            s=40,
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            traj[-1, 2],
            marker="x",
            s=60,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajectories")
    ax.legend()
    plt.show()

def main():
    #TODO: DO EVALUATION BETWEEN TWO ZARR DATASETS, THESE TWO ARE THE ESTIMATEN AND GROUND TRUTH

    window_size = 2000
    position_dim = 3

    trajectories = [
        {
            "path": "/media/egghead/Scratch/joey/simulation_data/patient_two_data.zarr/",
            "episode": 3,
            "label": "GT",
        },
        {
            "path": "/media/egghead/Scratch/joey/simulation_data/patient_three_data.zarr/",
            "episode": 3,
            "label": "EKF2",
        },
    ]

    plot_trajectories(
        trajectories=trajectories,
        pred_dim=position_dim,
    )


if __name__ == "__main__":
    main()
