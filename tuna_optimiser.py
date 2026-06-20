import optuna
import hydra
import torch
from omegaconf import OmegaConf

from include.mama import JeuralJetwork
from QuickDataset import QuickDataset2

from hydra_trainer import train_loop, train_rollout_loop
from batched_evaluator import evaluate_model   # your evaluation file

device = torch.device("cuda")

def make_train_loader(cfg):
    total_len = cfg.input_len + cfg.output_len

    dataset = QuickDataset2(
        path=cfg.dataset.path,
        training_size=cfg.dataset.training_size,
        window_size=total_len + cfg.training.rollout_steps,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=20,
        pin_memory=True,
        persistent_workers=True,
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


def objective(trial, base_cfg):
    cfg = OmegaConf.create(
        OmegaConf.to_container(
            base_cfg,
            resolve=True
        )
    )

    print("\n" + "="*60)
    print(f"STARTING TRIAL {trial.number}")
    print("="*60)

    cfg.models.architecture.d_model = trial.suggest_categorical(
        "d_model",
        [128,256,512]
    )

    cfg.models.architecture.n_encoder_layers = trial.suggest_int(
        "encoder_layers",
        1,
        5
    )

    cfg.models.architecture.block_type = trial.suggest_categorical(
        "block_type",
        ["simple", "advanced", "cls"]
    )

    cfg.models.architecture.cls_kwargs.n_heads = trial.suggest_categorical(
        "cls_heads",
        [2,4,8]
    )
    cfg.models.architecture.cls_kwargs.dropout = trial.suggest_float(
        "cls_dropout",
        0.0,
        0.3
    )

    cfg.training.lr = trial.suggest_float(
        "lr",
        5e-4,
        1e-3,
        log=True
    )

    cfg.input_len = trial.suggest_categorical(
        "input_len",
        [8,16,32,64,128]
    )

    cfg.training.weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-1,
        log=True
    )

    print("Hyperparameters:")
    print(
        f"""
        d_model        : {cfg.models.architecture.d_model}
        enc layers     : {cfg.models.architecture.n_encoder_layers}
        lr             : {cfg.training.lr:.2e}
        weight decay   : {cfg.training.weight_decay:.2e}
        input_len      : {cfg.input_len:.2e}
        block_type     : {cfg.models.architecture.block_type}
        """
    )

    model = JeuralJetwork(
        n_dim=cfg.models.n_dim,
        out_dim=cfg.models.out_dim,
        input_len=cfg.input_len,
        output_len=cfg.output_len,
        **cfg.models.architecture
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )

    loader = make_train_loader(cfg)
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed)

    for epoch in range(5):

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
        eps_indices=[0,1,2,3,4],
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

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=100,
        callbacks=[print_callback],
        n_jobs=2,
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

