import optuna
import hydra
import torch
import zarr
from omegaconf import OmegaConf

from include.mama import JeuralJetwork
from QuickDataset import QuickDataset2

from hydra_trainer import train_loop, train_rollout_loop
from batched_evaluator import evaluate_model   # your evaluation file

device = torch.device("cuda")
import copy

TRAIN_LOADER = None
ZARR_ROOT = None

def build_architecture(trial, cfg):
    arch = copy.deepcopy(cfg.models.architecture)

    # -----------------------
    # CORE MODEL CAPACITY
    # -----------------------
    arch["d_model"] = trial.suggest_categorical("d_model", [16, 32, 128])
    arch["n_encoder_layers"] = trial.suggest_int("enc_layers", 1, 3)

    arch["layer_scale"] = trial.suggest_float("layer_scale", 0.0, 1e-4)
    arch["drop_path"] = trial.suggest_float("drop_path", 0.0, 0.2)

    # -----------------------
    # BLOCK TYPE
    # -----------------------
    arch["block_type"] = trial.suggest_categorical(
        "block_type",
        ["simple", "advanced", "cls"]
    )

    # -----------------------
    # MAMBA BLOCK
    # -----------------------
    if arch["block_type"] in ["simple", "advanced"]:

        mk = arch["mamba_kwargs"]

        mk["d_state"] = trial.suggest_categorical("mamba_d_state", [8, 16, 32, 64])
        mk["expand"] = trial.suggest_categorical("mamba_expand", [2, 4])

        # IMPORTANT missing knobs from YAML
        mk["d_conv"] = trial.suggest_categorical("mamba_d_conv", [2, 4])

        mk["norm"] = trial.suggest_categorical("mamba_norm", ["none", "rms", "layer"])

        # optional if your model supports it safely
        # mk["dt_rank"] = trial.suggest_categorical("mamba_dt_rank", [16, 32])

        arch["cls_kwargs"] = None  # ensure no leakage

    # -----------------------
    # CLS BLOCK
    # -----------------------
    elif arch["block_type"] == "cls":

        ck = arch["cls_kwargs"]

        ck["n_heads"] = trial.suggest_categorical("cls_heads", [2, 4, 8])
        ck["dropout"] = trial.suggest_float("cls_dropout", 0.0, 0.3)

        ck["positional_encoding"] = trial.suggest_categorical(
            "cls_posenc",
            [True, False]
        )

        ck["max_len"] = trial.suggest_categorical(
            "cls_max_len",
            [512, 1000, 2000, 5000]
        )

        arch["mamba_kwargs"] = None  # ensure no leakage

    return arch

def print_architecture(trial, cfg, arch):
    print("\n" + "=" * 70)
    print(f"TRIAL {trial.number} CONFIG")
    print("=" * 70)

    print(f"""
block_type      : {arch.get("block_type")}
d_model         : {arch.get("d_model")}
enc_layers      : {arch.get("n_encoder_layers")}
drop_path       : {arch.get("drop_path")}
layer_scale     : {arch.get("layer_scale")}
input_len       : {cfg.input_len}
lr              : {cfg.training.lr:.2e}
weight_decay    : {cfg.training.weight_decay:.2e}
rollout_steps   : {cfg.training.rollout_steps}
""")

    # ---- MAMBA ----
    mk = arch.get("mamba_kwargs")
    if mk is not None:
        print("---- MAMBA ----")
        print(f"d_state        : {mk.get('d_state')}")
        print(f"d_conv         : {mk.get('d_conv')}")
        print(f"expand         : {mk.get('expand')}")
        print(f"norm           : {mk.get('norm')}")
        print(f"dt_rank        : {mk.get('dt_rank')}")

    # ---- CLS ----
    ck = arch.get("cls_kwargs")
    if ck is not None:
        print("---- CLS ----")
        print(f"n_heads        : {ck.get('n_heads')}")
        print(f"dropout        : {ck.get('dropout')}")
        print(f"pos_encoding   : {ck.get('positional_encoding')}")
        print(f"max_len        : {ck.get('max_len')}")

    print("=" * 70 + "\n")

def make_train_loader(cfg):

    max_input = 128
    max_rollout = 48

    total_len = (
        max_input
        + cfg.output_len
        + max_rollout
    )

    dataset = QuickDataset2(
        path=cfg.dataset.path,
        training_size=cfg.dataset.training_size,
        window_size=total_len,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=5,
    )

    return loader

@torch.no_grad()
def quick_validation_loss(model, loader):
    model.eval()

    total = 0
    count = 0
    for vec in loader:
        vec = vec.to(device)

        x = vec[:, :model.input_len, :]
        truth = (
            vec[:, model.input_len:model.input_len+1, :3]
            -
            vec[:, model.input_len-1:model.input_len, :3]
        )
        pred = model(x)
        loss = torch.nn.functional.mse_loss(
            pred[:, :, :3],
            truth
        )
        total += loss.item()
        count += 1

    return total/count


def objective(trial, base_cfg, loader, root):
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            base_cfg,
            resolve=True
        )
    )

    print("\n" + "="*60)
    print(f"STARTING TRIAL {trial.number}")
    print("="*60)

    cfg.training.lr = trial.suggest_float("lr",5e-4,1e-3,log=True)
    cfg.input_len = trial.suggest_categorical("input_len", [8,16,32,64,128])
    cfg.training.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    cfg.training.rollout_steps = trial.suggest_int("rollout_steps", 5, 48, log=True)

    arch = build_architecture(trial, cfg)
    print_architecture(trial, cfg, arch)

    model = JeuralJetwork(
        n_dim=cfg.models.n_dim,
        out_dim=cfg.models.out_dim,
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        **arch
    ).to(device)
    #model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    loader = make_train_loader(cfg)
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

    root = zarr.open(
        zarr.storage.LocalStore(cfg.evaluation.path),
        mode="r"
    )["episodes"]

    for epoch in range(3):

        print(
            f"Trial {trial.number} | epoch {epoch+1}/5"
        )

        train_rollout_loop(
            loader,
            model,
            optimizer,
            batch_size=cfg.batch_size,
            process_name=f"trial {trial.number}",
            generator=gen,
            rollout_max_steps=cfg.training.rollout_steps
        )

        if epoch > 0:
            mini_acc = evaluate_model(
                model,
                root,
                eps_indices=[0],
                input_len=cfg.input_len,
                output_len=cfg.output_len,
                device=device
            )
            val = mini_acc.average()["ate_rmse"]
        else:
            val = quick_validation_loss(
                model,
                loader
            )

        print(
            f"validation loss: {val:.6f}"
        )

        trial.report(
            val,
            epoch
        )

        if trial.should_prune():

            print(
                f"TRIAL {trial.number} PRUNED "
                f"(epoch {epoch})"
            )

            raise optuna.TrialPruned()

    print(
        f"Running trajectory evaluation for trial {trial.number}..."
    )

    acc = evaluate_model(
        model,
        root,
        eps_indices=[0,1,2],
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        device=device
    )

    metrics = acc.average()
    score = (
        metrics["ate_rmse"]
        + 0.1 * metrics["endpoint_error"]
        + 0.01 * metrics["drift_m_per_km"]
    )

    print(f"""
        TRIAL {trial.number} RESULT

        ATE RMSE       : {metrics["ate_rmse"]:.4f}
        Endpoint error : {metrics["endpoint_error"]:.4f}
        Drift m/km     : {metrics["drift_m_per_km"]:.4f}

        TOTAL SCORE    : {score:.4f}
        """
    )

    del model
    del optimizer
    torch.cuda.empty_cache()

    return score

def print_callback(study, trial):

    print(f"""
        Finished trial {trial.number}

        Current best:
        trial : {study.best_trial.number}
        score : {study.best_value:.5f}
        params:
        {study.best_params}
        """
    )

@hydra.main(
    version_base=None,
    config_path="hydra-cfgs",
    config_name="config"
)
def main(cfg):
    global TRAIN_LOADER
    global ZARR_ROOT

    print("Loading dataset once...")

    TRAIN_LOADER = make_train_loader(cfg)
    ZARR_ROOT = zarr.open(
        zarr.storage.LocalStore(
            cfg.evaluation.path
        ),
        mode="r"
    )["episodes"]

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        lambda trial: objective(
            trial,
            cfg,
            TRAIN_LOADER,
            ZARR_ROOT,
        ),
        n_trials=50,
        callbacks=[print_callback],
        n_jobs=5,
        )

    print("\n====================")
    print("OPTIMIZATION DONE")
    print("====================")

    print(
        "Best score:",
        study.best_value
    )

    print(
        "Best params:"
    )

    best_cfg = OmegaConf.create(cfg)

    for k,v in study.best_params.items():
        print(f"  {k}: {v}")

    OmegaConf.save(
        best_cfg,
        "best_config.yaml"
    )


if __name__ == "__main__":
    main()

