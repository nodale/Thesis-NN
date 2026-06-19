import numpy as np
import torch

def align_trajectory(pred, truth):
    """
    Align predicted trajectory to ground truth using Umeyama alignment.
    For odometry without scale drift, SE(3) is enough.
    """
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    truth = truth.cpu().numpy() if torch.is_tensor(truth) else truth

    mu_pred = pred.mean(axis=0)
    mu_truth = truth.mean(axis=0)
    X = pred - mu_pred
    Y = truth - mu_truth
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # reflection handling
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = mu_truth - R @ mu_pred
    aligned = (R @ pred.T).T + t
    return aligned

def metric_trajectory_length(x):
    x = x.cpu().numpy()
    return np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))

def metric_drift_rate(pred, truth):
    """
    Drift rate.
    Returns:
        percent drift
        meters per km
    """
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()

    final_error = np.linalg.norm(pred[-1]-truth[-1])
    distance = np.sum(
        np.linalg.norm(
            np.diff(truth, axis=0),
            axis=1
        )
    )
    percent = (final_error / distance * 100)
    m_per_km = (final_error / distance * 1000)
    return {"drift_percent": percent, "drift_m_per_km": m_per_km}

def metric_ate(pred, truth, align=True):
    """
    Absolute Trajectory Error RMSE
    Returns:
        rmse [meters]
    """
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    if align:
        pred = align_trajectory(pred, truth)
    error = pred - truth
    dist = np.linalg.norm(error, axis=1)
    return np.sqrt(np.mean(dist**2))

def metric_kitti_odometry(pred, truth, segment_lengths=None):
    """
    KITTI style odometry evaluation.
    Returns:
        translation drift %
        rotation drift deg/m
    Only works for XYZ trajectories.
    """
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    if segment_lengths is None:
        segment_lengths = [100,200,300,400,500,600,700,800]
    trans_errors = []
    rot_errors = []
    N = len(pred)
    for length in segment_lengths:
        step = length
        for i in range(N-step):
            gt_dist = np.linalg.norm(truth[i+step]-truth[i])
            if gt_dist < 1e-6:
                continue
            pred_delta = pred[i+step]-pred[i]
            gt_delta = truth[i+step]-truth[i]

            trans_error = (np.linalg.norm(pred_delta-gt_delta)/gt_dist)
            trans_errors.append(trans_error*100)
            # rotation not available from xyz only
    return {"translation_error_percent": np.mean(trans_errors), "rotation_error_deg_per_meter": None}

def metric_endpoint_error(pred, truth):
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    return np.linalg.norm(pred[-1]-truth[-1])

def metric_mean_error(pred, truth):
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    return np.mean(np.linalg.norm(pred-truth, axis=1))

def metric_max_error(pred, truth):
    pred = pred.cpu().numpy()
    truth = truth.cpu().numpy()
    return np.max(np.linalg.norm(pred-truth, axis=1))

def print_all_metrics(predicted, truth):
    print("\n===== Odometry Metrics =====")
    print("trajectory length:", metric_trajectory_length(truth), "m")
    print("ATE RMSE:", metric_ate(predicted, truth, align=True), "m")

    print("Mean error:", metric_mean_error(predicted, truth), "m")
    print("Max error:", metric_max_error(predicted, truth), "m")
    print("Endpoint error:", metric_endpoint_error(predicted, truth), "m")

    kitti = metric_kitti_odometry(predicted, truth)
    print("KITTI translation drift:", kitti["translation_error_percent"], "%")

    drift = metric_drift_rate(predicted, truth)
    print("Drift:", drift["drift_percent"], "%", "(", drift["drift_m_per_km"], "m/km )")

