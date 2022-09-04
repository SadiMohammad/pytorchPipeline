import time
import torch
import os, sys, random, string
import argparse
import wandb
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms
from models.model import Model
from configs.config import Config
from utils.logger import Logger
from dataloaders.dataloader import Dataset_ROM, Dataset_RAM
from trainer import Trainer

time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")


def main(Cfgs):
    cfgs = Cfgs.cfgs
    device = torch.device(cfgs["train_setup"]["device"])

    # LOGGING
    if cfgs["logs"]["save_local_logs"]:
        sys.stdout = Logger(
            os.path.join(
                cfgs["logs"]["local_logs_path"],
                cfgs["experiment_name"],
                "{}.log".format(time_stamp),
            )
        )
    if cfgs["logs"]["use_wandb"]:
        run_name = (
            cfgs["experiment_name"]
            + "-"
            + "".join(random.choices(string.ascii_lowercase, k=5))
        )
        wandb.init(
            project=cfgs["logs"]["pytorchPipeline"],
            entity=cfgs["logs"]["wandb_entity"],
            config=Cfgs.cfgs_single_dict,
            name=run_name,
        )

    # DATA LOADERS
    transformers = {
        "image": transforms.Compose(
            [
                transforms.Resize((cfgs["dataset"]["input_size"], cfgs["dataset"]["input_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.485, std=0.229),
            ]
        ),
        "gt": transforms.Compose(
            [
                transforms.Resize((cfgs["dataset"]["input_size"] cfgs["dataset"]["input_size"])),
                transforms.ToTensor(),
            ]
        ),
    }
    dataset_train = Dataset_ROM(cfgs, transformers=transformers)
    loader_train = DataLoader(
        dataset_train,
        batch_size=cfgs["train_setup"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    dataset_valid = Dataset_ROM(cfgs, transformers=transformers)
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfgs["train_setup"]["batch_size"],
        shuffle=False,
        pin_memory=True,
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
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfgs["optimizer"]["optimizer_fn"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )
    loss_fn = cfgs["train_setup"]["loss"]
    metric_fn = cfgs["train_setup"]["metric"]
    Trainer(
        time_stamp,
        model,
        optimizer,
        device,
        loader_train,
        loader_valid,
        loss_fn,
        metric_fn,
    )


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
