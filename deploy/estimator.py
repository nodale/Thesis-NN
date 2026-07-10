"""
estimator.py

Loads a trained JeuralJetwork odometry model from a Hydra run (the same
config/checkpoint layout produced by the training code and consumed by
eval.py) and runs it *online* inside a vectorized IsaacLab simulation.

Relationship to eval.py
------------------------
eval.py's `evaluate()` runs one trajectory at a time, offline, from a
DataLoader: it keeps a rolling window `init_pos` of the model's own
predicted position (not ground truth), autoregressively rolls it forward
one step at a time, and denormalises only at the very end for metrics.

`QuickEstimator` below does the same *algorithm*, but:
  - batched over the environment dimension (num_envs) instead of iterating
    a single trajectory, since IsaacLab steps every env in lockstep
  - called incrementally (`.step(obs)` once per control tick) instead of
    iterating over a pre-built windowed dataset
  - supports partial resets, since IsaacLab only resets a subset of envs
    at a time (see DirectRLEnv._reset_idx / DroneEnv._reset_idx)

Why the model's own prediction is fed back in (not ground truth)
------------------------------------------------------------------
This mirrors the Legolas-style deployment setup this model is trained
under: position/velocity are exactly the things being estimated, so a
real deployment never has ground truth for those channels available to
feed back into the model. eval.py replicates this with
`_in[:input_len, :pred_dim] = init_pos` inside its loop; this module does
the analogous overwrite of the leading `pred_dim` columns of each new
observation with the running estimate before pushing it into history.
The other channels (attitude, angular rate, IMU, thrust, setpoint) are
still taken directly from simulation/sensors each step, exactly as in
DroneEnv._pre_physics_step, since those are already directly measured
and are not part of what the network is estimating.

This file intentionally does not modify environment_isaac.py. It is
meant to be instantiated once in DroneEnv.__init__ (after `self.n_dim`
etc. are known) and driven from `_pre_physics_step` / `_reset_idx`. See
the usage example at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import yaml

from model.network import JeuralJetwork
from data.denormaliser import denormalise


# ----------------------------------------------------------------------
# Hydra run loading helpers (mirrors get_latest_multirun / load_run /
# load_model in eval.py)
# ----------------------------------------------------------------------

def _load_hydra_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No Hydra config found at {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _find_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("**/*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoint found under {run_dir}")
    return ckpts[-1]


def _strip_compile_prefix(state_dict: dict) -> dict:
    # torch.compile() wraps parameters under "_orig_mod." - strip it so
    # the state dict loads cleanly onto an uncompiled module, then let
    # cfg["compile"] decide whether to re-wrap after loading (same order
    # of operations as load_model() in eval.py).
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _build_model(cfg: dict) -> torch.nn.Module:
    # NOTE: eval.py has two slightly different out_dim conventions between
    # load_run() (doubles out_dim unless mode == "rollout", for a
    # mean+variance head) and load_model() (uses out_dim as-is). This
    # follows load_run()'s convention since that's what the sweep-evaluation
    # entrypoint in eval.py actually exercises end-to-end. If your
    # checkpoint was produced/consumed via load_model()'s convention
    # instead, pass a pre-built model into QuickEstimator's constructor
    # directly rather than using `from_run`.
    mode = cfg["training"]["mode"]
    out_dim = cfg["models"]["out_dim"]
    if mode != "rollout":
        out_dim *= 2

    return JeuralJetwork(
        n_dim=cfg["models"]["n_dim"],
        out_dim=out_dim,
        input_len=cfg["input_len"],
        output_len=cfg["output_len"],
        **cfg["models"]["architecture"],
    )


class QuickEstimator:
    """
    Batched, online wrapper around a trained JeuralJetwork odometry model,
    covering all `num_envs` parallel IsaacLab environments with a single
    shared set of weights (one batched forward call per control tick,
    rather than `num_envs` separate model instances - the model doesn't
    need per-env weights, only per-env history/estimate state, which is
    what this class actually keeps separate per environment).

    Typical usage inside DroneEnv (illustrative only - not wired in here):

        # in __init__, after self.n_dim / self.scene.num_envs are known:
        self.estimator = QuickEstimator.from_run(
            run_dir=Path("multirun/2026-07-08/12-00-00/0"),
            num_envs=self.scene.num_envs,
            device=self.device,
            pred_dim=6,   # e.g. position (3) + linear velocity (3)
        )

        # in _pre_physics_step, once the per-env `obs` feature vector
        # (pos, lin_vel, quat_frd, ang_vel, acc, thrust, setpoint) has
        # been assembled, in the same channel order the model was
        # trained on:
        est_state = self.estimator.step(obs)   # None until history fills
        if est_state is not None:
            states_for_ctrl = self.estimator.build_controller_states(
                est_state, quat_frd, ang_vel
            )
            self.drone.controller.step(
                desired_pos=setpoint,
                states=states_for_ctrl,
                dt=1.0 / self.sampling_freq,
                physics_dt=self.cfg.sim.dt,
            )

        # in _reset_idx, alongside the other per-env resets:
        self.estimator.reset(env_ids)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict,
        num_envs: int,
        device: Union[torch.device, str],
        pred_dim: int = 6,
    ):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = num_envs

        self.input_len: int = cfg["input_len"]
        self.output_len: int = cfg["output_len"]
        self.n_dim: int = cfg["models"]["n_dim"]
        self.pred_dim = pred_dim

        if pred_dim > self.n_dim:
            raise ValueError(
                f"pred_dim ({pred_dim}) cannot exceed the model's n_dim ({self.n_dim})"
            )

        # Rolling per-env observation history, in normalised units (the
        # same units the model was trained on) - shape (num_envs, input_len, n_dim).
        self.history = torch.zeros(
            num_envs, self.input_len, self.n_dim, device=self.device
        )
        # Count of valid (non-padding) steps pushed per env, capped at
        # input_len - lets .step() know when a given env's window is
        # actually full enough to trust a prediction from.
        self.filled = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Running, normalised position/velocity estimate per env, updated
        # autoregressively from the model's own predicted deltas - this
        # is the per-env analogue of eval.py's `init_pos`, except it
        # persists across calls instead of being (re-)initialised from
        # the first `input_len` ground-truth rows of a trajectory.
        self.position_estimate = torch.zeros(num_envs, pred_dim, device=self.device)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_run(
        cls,
        run_dir: Union[str, Path],
        num_envs: int,
        device: Union[torch.device, str],
        pred_dim: int = 6,
    ) -> "QuickEstimator":
        """Build from a single Hydra run directory (containing .hydra/config.yaml
        and a .pth checkpoint somewhere under it), the same layout load_run()
        reads in eval.py."""
        run_dir = Path(run_dir)
        cfg = _load_hydra_cfg(run_dir)
        ckpt_path = _find_checkpoint(run_dir)

        model = _build_model(cfg)
        state = torch.load(ckpt_path, map_location=device)
        state = _strip_compile_prefix(state)
        model.load_state_dict(state)
        model.to(device)
        if cfg.get("compile", False):
            model = torch.compile(model)
        model.eval()

        return cls(model=model, cfg=cfg, num_envs=num_envs, device=device, pred_dim=pred_dim)

    @classmethod
    def from_latest_multirun(
        cls,
        multirun_root: Union[str, Path],
        num_envs: int,
        device: Union[torch.device, str],
        pred_dim: int = 6,
    ) -> "QuickEstimator":
        """Convenience constructor mirroring get_latest_multirun()/latest_sweep()
        in eval.py - picks the most-recently-modified Hydra run under a
        `multirun/` tree and loads it."""
        multirun_root = Path(multirun_root)
        runs = [p for p in multirun_root.glob("*/*") if (p / ".hydra").exists()]
        if not runs:
            raise RuntimeError(f"No Hydra runs with a .hydra dir found under {multirun_root}")
        latest = max(runs, key=lambda p: p.stat().st_mtime)
        return cls.from_run(latest, num_envs=num_envs, device=device, pred_dim=pred_dim)

    # ------------------------------------------------------------------
    # runtime
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def step(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Push one new (normalised) observation per environment into the
        rolling history, and - once every env's window is full - run one
        batched forward pass and update the per-env position estimate.

        Args:
            obs: (num_envs, n_dim) tensor, in the same normalised feature
                layout as self.data in DroneEnv (pos/3.0, lin_vel/0.8,
                quat_frd, ang_vel/0.5, acc/25.0, thrust/9.81, setpoint/3.0,
                ...), for the *current* timestep. The leading `pred_dim`
                columns (whatever the model was trained to predict, e.g.
                position and/or velocity) will be overwritten in-place
                with this estimator's own running estimate before being
                pushed into history - see module docstring for why.

        Returns:
            (num_envs, pred_dim) tensor of the updated position/velocity
            estimate, denormalised back to physical units - or None if at
            least one environment's history window is not yet full
            (equivalent to eval.py needing `input_len` steps of history
            before evaluate() produces its first prediction).
        """
        if obs.shape != (self.num_envs, self.n_dim):
            raise ValueError(
                f"expected obs of shape ({self.num_envs}, {self.n_dim}), got {tuple(obs.shape)}"
            )
        obs = obs.to(self.device).clone()

        # Replace the channels the model is responsible for estimating
        # with our own running (normalised) estimate rather than whatever
        # ground-truth/sim value the caller may have put there - keeps
        # inference consistent with a real deployment. Equivalent to
        # eval.py's `_in[:input_len, :pred_dim] = init_pos`.
        obs[:, : self.pred_dim] = self.position_estimate

        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = obs
        self.filled = torch.clamp(self.filled + 1, max=self.input_len)

        if bool((self.filled < self.input_len).any()):
            return None

        out = self.model(self.history)              # (num_envs, output_len, out_dim)
        delta = out[:, 0, : self.pred_dim]           # first predicted step, mean component only

        self.position_estimate = self.position_estimate + delta
        return denormalise(self.position_estimate.detach().cpu()).to(self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear history/estimate for a subset of environments. Call this
        alongside the rest of DroneEnv._reset_idx's per-env resets so a
        newly-reset env doesn't drag stale history/estimate into its next
        episode."""
        if env_ids is None or len(env_ids) == 0:
            return
        self.history[env_ids] = 0.0
        self.filled[env_ids] = 0
        self.position_estimate[env_ids] = 0.0

    def seed_estimate(self, env_ids: torch.Tensor, normalised_state: torch.Tensor) -> None:
        """Optionally seed the running (normalised) estimate for a subset
        of envs - e.g. from the true simulator state right after a reset,
        rather than zero - to avoid a transient at the start of each
        episode while ground truth is actually available for free in sim.
        Not used by default; call explicitly if desired."""
        self.position_estimate[env_ids] = normalised_state.to(self.device)

    @property
    def is_ready(self) -> torch.Tensor:
        """(num_envs,) bool tensor: whether each env's history window is
        currently full enough to have produced a real prediction."""
        return self.filled >= self.input_len

    @staticmethod
    def build_controller_states(
        estimated: torch.Tensor,
        quat: torch.Tensor,
        ang_vel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble the state vector Kxontroller.step(states=...) expects,
        by combining the network's estimated channels (e.g. position +
        linear velocity) with attitude/rate channels taken directly from
        the IMU/simulator, which are already directly measured and are
        not part of what this model estimates.

        `estimated` is whatever QuickEstimator.step() returned this
        tick (num_envs, pred_dim). Adjust the slicing here if pred_dim
        covers a different set of channels than position+velocity for
        your particular trained checkpoint.
        """
        return torch.cat([estimated, quat, ang_vel], dim=1)


if __name__ == "__main__":
    """
    Example: load a real JeuralJetwork from the latest Hydra multirun and
    run a handful of simulated step() calls to verify the pipeline.

    Usage:
        python -m deploy.estimator
        python -m deploy.estimator --run multirun/2026-07-08/12-00-00/0
        python -m deploy.estimator --envs 8 --pred-dim 6
    """
    import argparse

    parser = argparse.ArgumentParser(description="StateEstimator load/step smoke test")
    parser.add_argument(
        "--run",
        default=None,
        help="Path to a specific Hydra run directory. "
             "Omit to use the most recently modified run under --multirun-root.",
    )
    parser.add_argument(
        "--multirun-root",
        default="multirun",
        help="Root directory to search for Hydra runs when --run is not given. "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=2,
        help="Number of parallel environments to simulate. Default: %(default)s",
    )
    parser.add_argument(
        "--pred-dim",
        type=int,
        default=6,
        help="Leading channels of the observation the model predicts "
             "(e.g. 3 for position-only, 6 for position+lin_vel). Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Default: cuda if available, else cpu.",
    )
    args = parser.parse_args()

    if args.run is not None:
        run_dir = Path(args.run)
        print(f"Loading from specified run: {run_dir}")
        estimator = QuickEstimator.from_run(
            run_dir=run_dir,
            num_envs=args.envs,
            device=args.device,
            pred_dim=args.pred_dim,
        )
    else:
        print(f"Searching for latest Hydra run under: {args.multirun_root}")
        estimator = QuickEstimator.from_latest_multirun(
            multirun_root=args.multirun_root,
            num_envs=args.envs,
            device=args.device,
            pred_dim=args.pred_dim,
        )

    cfg = estimator.cfg
    print(
        f"\nLoaded JeuralJetwork:"
        f"\n  n_dim      = {cfg['models']['n_dim']}"
        f"\n  out_dim    = {cfg['models']['out_dim']}  "
        f"(doubled to {cfg['models']['out_dim'] * 2} for mean+variance "
        f"unless mode == 'rollout')"
        f"\n  input_len  = {cfg['input_len']}"
        f"\n  output_len = {cfg['output_len']}"
        f"\n  architecture = {cfg['models']['architecture']}"
        f"\n  device     = {args.device}"
        f"\n  num_envs   = {args.envs}"
        f"\n  pred_dim   = {args.pred_dim}"
    )
    print(f"\n{estimator.model}\n")

    n_dim = cfg["models"]["n_dim"]
    input_len = cfg["input_len"]
    total_steps = input_len + 3

    print(f"Stepping {total_steps} ticks ({input_len} to fill window, then 3 live predictions):\n")
    for t in range(total_steps):
        fake_obs = torch.randn(args.envs, n_dim, device=args.device)
        result = estimator.step(fake_obs)

        if result is None:
            print(f"  t={t:3d}  history filling  filled={estimator.filled.tolist()}")
        else:
            print(
                f"  t={t:3d}  estimate ready   "
                f"shape={tuple(result.shape)}  "
                f"mean_norm={result.norm(dim=1).mean().item():.4f}"
            )

    print(f"\nResetting env 0 only ...")
    estimator.reset(torch.tensor([0], device=args.device))
    print(f"  filled after reset: {estimator.filled.tolist()}")
    print(f"  is_ready:           {estimator.is_ready.tolist()}")
    print("\nDone.")
