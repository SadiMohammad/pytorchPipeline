import time
import torch
import os, sys, random, string
import argparse
import wandb
from torch.utils.data import DataLoader
from torch import optim
from models.model import Model
from configs.config import Config
from utils.logger import Logger
from dataloaders.dataloader import Dataset_ROM, Dataset_RAM
from trainer import Trainer
from utils.save_config import RunHistory
from utils.transforms import get_transformers

time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")


def main(Cfgs):
    cfgs = Cfgs.cfgs
    device = torch.device(cfgs["train_setup"]["device"])

    # LOGGING
    if cfgs["logs"]["save_local_logs"]:
        log_dir = os.path.join(
            cfgs["logs"]["local_logs_path"],
            cfgs["experiment_name"],
        )
        if not (os.path.exists(log_dir)):
            os.makedirs(log_dir)
        sys.stdout = Logger(os.path.join(log_dir, "{}.log".format(time_stamp)))

    if cfgs["logs"]["save_local_config"]:
        RunHistory(time_stamp, cfgs, cfgs["logs"]["local_cfgs_path"]).save_run_history()

    if cfgs["logs"]["use_wandb"]:
        run_name = (
            cfgs["experiment_name"]
            + "-"
            + "".join(random.choices(string.ascii_lowercase, k=5))
        )
        wandb.init(
            project=cfgs["logs"]["wandb_project_name"],
            entity=cfgs["logs"]["wandb_entity"],
            config=Cfgs.cfgs_single_dict,
            name=run_name,
        )

    # DATA LOADERS
    transformers = get_transformers(cfgs)
    dataset_train = Dataset_ROM(cfgs, subset="train", transformers=transformers)
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfgs["train_setup"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    dataset_valid = Dataset_ROM(cfgs, subset="val", transformers=transformers)
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfgs["train_setup"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # MODEL
    model = Model(cfgs).get_model()
    model = model.to(device)
    if cfgs["logs"]["use_wandb"]:
        wandb.watch(model)

    # Training
    optimizer = getattr(optim, cfgs["optimizer"]["optimizer_fn"])(
        model.parameters(),  # momentum=0.9,  # for sgd
        lr=cfgs["optimizer"]["initial_lr"],
        weight_decay=0.0005,
    )

    loss_fn = cfgs["train_setup"]["loss"]
    metric_fn = cfgs["train_setup"]["metric"]

    trainer = Trainer(
        cfgs=cfgs,
        time_stamp=time_stamp,
        model=model,
        optimizer=optimizer,
        device=device,
        loader_train=loader_train,
        loader_valid=loader_valid,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
    )
    trainer.train()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        help="config file name/experiment name",
        default="train",
    )
    args = arg_parser.parse_args()
    Cfgs = Config(args)
    main(Cfgs)
