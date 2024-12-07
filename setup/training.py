import argparse
import random
import numpy
import torch
import wandb
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from chemprop import data, featurizers, models, nn

# TODO: Do this in a more automatic way maybe or add them to parser
# TODO: Maybe add to parser the wandb project name
# TODO: Also need to try and integrate this in the gflownet mpnn inference
# TODO: I also need how to create a repo and import 2 forked ones

INPUT_PATH = "data/KOW.csv"
SMILES_COL = "smiles"
TARGET_COL = ["active"]
# These are not used in hyperparam tuning for now
SPLIT_TYPE = "SCAFFOLD_BALANCED"
SPLIT_SIZE = [0.8, 0.1, 0.1]

# TODO: When used for gpu training change the project name to create new one
# TODO: Add more hyperparams for hidden dims maybe
# TODO: Check how to add learning rate and optimizer (current)

"""python training.py --scaling False --batch_size 248 --depth 2 --dropout 0.2 --acti
vation_mpnn 'relu' --aggregation 'mean' --hidden_dim_readout 64
--hidden_layers_readout 1 --dropout_readout 0.2 --batch_norm False
"""


def set_seed():
    random.seed(42)
    numpy.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def setup_run(args):
    wandb.login()
    wandb.init(
        project="New Test project",
        config={
            "scaling": args.scaling,
            "batch_size": args.batch_size,
            "depth": args.depth,
            "dropout": args.dropout,
            "activation_mpnn": args.activation_mpnn,
            "aggregation": args.aggregation,
            "hidden_dim_readout": args.hidden_dim_readout,
            "hidden_layers_readout": args.hidden_layers_readout,
            "dropout_readout": args.dropout_readout,
            "batch_norm": args.batch_norm,
            "init_lr": args.init_lr,
            "max_lr": args.max_lr,
            "final_lr": args.final_lr,
        },
    )


class WandbLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            "train/loss": trainer.callback_metrics.get("val_loss"),
            "train/rmse": trainer.callback_metrics.get("val_rmse"),
            "train/mae": trainer.callback_metrics.get("val_mae"),
            "train/r2": trainer.callback_metrics.get("val_r2_score"),
        }
        wandb.log(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            "val/loss": trainer.callback_metrics.get("val_loss"),
            "val/rmse": trainer.callback_metrics.get("val_rmse"),
            "val/mae": trainer.callback_metrics.get("val_mae"),
            "val/r2": trainer.callback_metrics.get("val_r2_score"),
        }
        wandb.log(metrics)


def ready_parser():
    parser = argparse.ArgumentParser(description="Train a MPNN model")
    parser.add_argument(
        "--scaling", type=bool, help="Whether to apply scaling on target", default=True
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training-val", default=64
    )
    parser.add_argument("--depth", type=int, help="Depth of MPNN", default=3)
    parser.add_argument("--dropout", type=float, help="Dropout for MPNN", default=0.1)
    parser.add_argument(
        "--activation_mpnn",
        type=str,
        help="Activation function for MPNN",
        default="relu",
    )
    parser.add_argument(
        "--aggregation", type=str, help="Aggregation function", default="mean"
    )
    parser.add_argument(
        "--hidden_dim_readout", type=int, help="Hidden dim for readout", default=128
    )
    parser.add_argument(
        "--hidden_layers_readout", type=int, help="Hidden layers for readout", default=1
    )
    parser.add_argument("--dropout_readout", type=float, help="Dropout for readout")
    parser.add_argument("--batch_norm", type=bool, help="Batch norm", default=False)
    parser.add_argument(
        "--init_lr", type=float, help="Initial learning rate", default=1e-3
    )
    parser.add_argument("--max_lr", type=float, help="Max learning rate", default=1e-2)
    parser.add_argument(
        "--final_lr", type=float, help="Final learning rate", default=1e-4
    )
    return parser.parse_args()


def load_data(input_path, smiles_col, target_col):
    df = pd.read_csv(input_path)
    smiles = df.loc[:, smiles_col].values
    target = df.loc[:, target_col].values

    all_data = [
        data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, target)
    ]

    return all_data


# TODO: Check atom_dim, hidden_dim, bond_dim
def get_activation(fn: str):
    if fn == "relu":
        return nn.Activation.RELU
    elif fn == "leaky_relu":
        return nn.Activation.LEAKYRELU
    elif fn == "prelu":
        return nn.Activation.PRELU
    elif fn == "selu":
        return nn.Activation.SELU
    elif fn == "elu":
        return nn.Activation.ELU
    else:
        raise ValueError(f"Activation function {fn} not supported")


def get_split(all_data: list, split_type: str, split_size: tuple[float, float, float]):
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, split_type, split_size
    )  # unpack the tuple into three separate lists
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    return train_data, val_data, test_data


def create_datasets(
    train_data: list, val_data: list, test_data: list, scaling: bool = False
):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(train_data[0], featurizer)
    val_dataset = data.MoleculeDataset(val_data[0], featurizer)
    if scaling:
        scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(scaler)
    else:
        scaler = None
    test_dataset = data.MoleculeDataset(test_data[0], featurizer)

    return train_dataset, val_dataset, test_dataset, scaler


def create_dataloader(
    train_dataset, val_dataset, test_dataset, batch_size, num_workers
):
    train_loader = data.build_dataloader(train_dataset, batch_size, num_workers)

    val_loader = data.build_dataloader(val_dataset, batch_size, num_workers)

    test_loader = data.build_dataloader(test_dataset, batch_size, num_workers)

    return train_loader, val_loader, test_loader


def message_passing(args):
    activation = get_activation(args.activation_mpnn)
    return nn.AtomMessagePassing(
        depth=args.depth, dropout=args.dropout, activation=activation
    )


def get_aggregation(aggr: str):
    if aggr == "mean":
        return nn.MeanAggregation()
    elif aggr == "sum":
        return nn.SumAggregation()
    else:
        raise ValueError(f"Aggregation function {aggr} not supported")


def readout_mlp(
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    scaling: bool = False,
    scaler: bool = None,
):
    if scaling:
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    else:
        output_transform = scaler
    return nn.RegressionFFN(
        hidden_dim=hidden_dim,
        n_layers=hidden_layers,
        dropout=dropout,
        output_transform=output_transform,
    )


def build_mpnn(mp, agg, ffn, args):
    # TODO: Fix this
    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=args.batch_norm,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()],
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
    )


def ready_trainer(max_epochs: int = 100):
    checkpointing = ModelCheckpoint(
        dirpath="checkpoints",  # Directory where model checkpoints will be saved
        filename="best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        monitor="val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
    )
    wandb_logger = WandbLoggingCallback()
    # TODO: Check optimizer, learning rate, weight decay, etc...
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,  # number of epochs to train for
        callbacks=[
            checkpointing,
            wandb_logger,
        ],  # Use the configured checkpoint callback
    )
    return trainer


def get_best_checkpoint(trainer: pl.Trainer):
    return trainer.checkpoint_callback.best_model_path


def load_model(checkpoint: str):
    return models.MPNN.load_from_checkpoint(checkpoint)


def rename_metrics(metrics: dict, split: str):
    metrics[split + "/mae"] = metrics.pop("val/mae")
    metrics[split + "/rmse"] = metrics.pop("val/rmse")
    metrics[split + "/r2"] = metrics.pop("val/r2")
    return metrics


def infer(train_loader, val_loader, test_loader, model: models.MPNN):
    # TODO: Load the config of train featurizer
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None, enable_progress_bar=True, accelerator="cpu", devices=1
        )
    train_metrics = trainer.validate(model, train_loader)[0]
    train_metrics = rename_metrics(train_metrics, "train")
    wandb.log(train_metrics)
    val_metrics = trainer.validate(model, val_loader)[0]
    val_metrics = rename_metrics(val_metrics, "val")
    wandb.log(val_metrics)
    test_metrics = trainer.validate(model, test_loader)[0]
    test_metrics = rename_metrics(test_metrics, "test")
    wandb.log(test_metrics)


def main():
    args = ready_parser()
    set_seed()
    setup_run(args)
    all_data = load_data(INPUT_PATH, SMILES_COL, TARGET_COL)
    train_data, val_data, test_data = get_split(all_data, SPLIT_TYPE, SPLIT_SIZE)
    train_dataset, val_dataset, test_dataset, scaler = create_datasets(
        train_data, val_data, test_data, args.scaling
    )
    train_loader, val_loader, test_loader = create_dataloader(
        train_dataset, val_dataset, test_dataset, args.batch_size, 0
    )
    mp = message_passing(args)
    agg = get_aggregation(args.aggregation)
    ffn = readout_mlp(
        args.hidden_dim_readout,
        args.hidden_layers_readout,
        args.dropout_readout,
        args.scaling,
        scaler,
    )
    model = build_mpnn(mp, agg, ffn, args)
    trainer = ready_trainer(max_epochs=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)
    ckpt = get_best_checkpoint(trainer)
    best_model = load_model(ckpt)
    infer(train_loader, val_loader, test_loader, best_model)
    wandb.finish()


if __name__ == "__main__":
    main()


"""python training.py --scaling False --batch_size 248 --depth 2 --dropout 0.2 --activation_mpnn 'relu' --aggregation 'mean' --hidden_dim_readout 64
--hidden_layers_readout 1 --dropout_readout 0.2 --batch_norm False --init_lr 0.0001 --max_lr 0.001 --final_lr 0.00001
"""
