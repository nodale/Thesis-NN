import sys
sys.path.insert(0, "/home/egghead/Thesis/Thesis-NN")

import torch
import zarr
import numpy as np

from include.metrics import MetricsAccumulator
from include.central_denormaliser import denormalise


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------

def load_episode_tensor(root, eps_indices):
    """
    Loads a batch of episodes.

    Returns
    -------
    Tensor [B, T, C]
    """

    episodes = []

    for ep in eps_indices:
        episodes.append(root[ep].astype(np.float32))

    return torch.from_numpy(np.stack(episodes))


# --------------------------------------------------
# Batched evaluation
# --------------------------------------------------

def evaluate_dataset_pair(
    gt_root,
    est_root,
    eps_indices,
    batch_size=32,
    pred_dim=6,
    denormalise_data=True,
):
    """
    Parameters
    ----------
    gt_root
        Ground-truth zarr["episodes"]

    est_root
        Estimated trajectory zarr["episodes"]

    Episode i from gt_root is compared ONLY against
    episode i from est_root.
    """

    acc = MetricsAccumulator()

    for start in range(0, len(eps_indices), batch_size):

        batch_eps = eps_indices[start:start + batch_size]

        # ----------------------------
        # load corresponding episodes
        # ----------------------------

        gt_batch = load_episode_tensor(gt_root, batch_eps)
        est_batch = load_episode_tensor(est_root, batch_eps)

        if gt_batch.shape != est_batch.shape:
            raise RuntimeError(
                f"Episode shape mismatch.\n"
                f"Ground truth : {gt_batch.shape}\n"
                f"Estimated    : {est_batch.shape}"
            )

        B = gt_batch.shape[0]

        for b in range(B):

            # ----------------------------
            # Ground Truth
            # ----------------------------
            truth = gt_batch[b, :, :pred_dim]

            # ----------------------------
            # Estimated trajectory
            # ----------------------------
            estimate = est_batch[b, :, :pred_dim]

            if denormalise_data:
                truth = denormalise(truth)
                estimate = denormalise(estimate)

            acc.update(
                estimate,
                truth,
            )

    return acc


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    ground_truth_path = "/media/egghead/Scratch/joey/simulation_data/patient_two_data.zarr/"
    estimated_path = "/media/egghead/Scratch/joey/simulation_data/patient_three_data.zarr/"

    gt_root = zarr.open(
        zarr.storage.LocalStore(ground_truth_path),
        mode="r",
    )["episodes"]

    est_root = zarr.open(
        zarr.storage.LocalStore(estimated_path),
        mode="r",
    )["episodes"]

    if gt_root.shape[0] != est_root.shape[0]:
        raise RuntimeError(
            f"Dataset size mismatch "
            f"({len(gt_root)} vs {len(est_root)})"
        )

    eps_indices = list(range(gt_root.shape[0]))

    acc = evaluate_dataset_pair(
        gt_root=gt_root,          # Ground Truth
        est_root=est_root,        # Estimated
        eps_indices=eps_indices,
        batch_size=64,
        pred_dim=6,
    )

    print("\n========================")
    print("Trajectory Evaluation")
    print("========================")
    acc.print()


if __name__ == "__main__":
    main()
