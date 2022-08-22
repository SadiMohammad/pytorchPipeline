import os
import torch
import wandb
from tqdm import tqdm
from utils.loss import Loss
from utils.metric import Metric


class Trainer:
    def __init__(
        self, **kwargs
    ):  # time_stamp, model, optimizer, device, loader_train, loader_valid, loss_fn, metric_fn
        self.__dict__.update(kwargs)

    def train(self):
        self.optimizer.zero_grad()
        best_valid_score = self.cfgs["train_setup"]["best_valid_score"]
        for epoch in range(self.cfgs["train_setup"]["epochs"]):
            print("\tStarting epoch - {}/{}.".format(epoch + 1, self.epochs))
            self.model.train()
            total_batch_loss = 0

            for i_train, sample_train in enumerate(tqdm(self.loader_train)):
                images = sample_train[0].to(self.device)
                labels = sample_train[1].to(self.device)
                preds = self.model(images)
                if self.cfgs["model"]["model_name"].lower() == "deeplabv3":
                    preds = preds["out"]
                    preds = torch.sigmoid(preds)

                m_batch_loss = getattr(Loss(labels, preds), self.loss_fn)()
                m_batch_metric = torch.mean(
                    getattr(Metric(labels, preds), self.metric_fn)()
                )
                total_batch_loss+=(torch.mean(m_batch_loss)).item()

                m_batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_validation_metric = self._validate(epoch)
            if self.cfgs["logs"]["use_wandb"]:
                wandb.log(
                    {
                        "epoch_mean_train_loss": total_batch_loss/(i_train+1),
                        "epoch_validation_metric": epoch_validation_metric,
                        "mini_batch_metric": m_batch_metric.item(),
                    }
                )
            if epoch_validation_metric > best_valid_score:
                best_valid_score = epoch_validation_metric
                if self.cfgs["train_setup"]["save_best_model"]:
                    save_ckpts = {
                        "epoch": epoch,
                        "input_size": self.input_size,
                        "best_score": best_valid_score,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "model_state_dict": self.model.state_dict(),
                    }
                    torch.save(
                        save_ckpts,
                        os.path.join(
                            self.cfgs["train_setup"]["checkpoints_path"],
                            self.cfgs["model"]["model_name"],
                            "{}.pth".format(self.time_stamp),
                        ),
                    )
                    print("!!! Checkpoint {} saved !!!".format(epoch + 1))

    def _validate(self, epoch):
        self.model.eval()
        epoch_valid_metric = 0
        with torch.no_grad():
            for i_valid, sample_valid in enumerate(self.loader_valid):
                images = sample_valid[0].to(self.device)
                labels = sample_valid[1].to(self.device)
                preds = self.model(images)
                if self.cfgs["model"]["model_name"].lower() == "deeplabv3":
                    preds = preds["out"]
                    preds = torch.sigmoid(preds)

                m_batch_loss = getattr(Loss(labels, preds), self.loss_fn)()
                m_batch_metric = torch.mean(
                    getattr(Metric(labels, preds), self.metric_fn)()
                )
                if self.cfgs["train_setup"]["use_thld_for_valid"]:
                    preds = (preds > self.cfgs["train_setup"]["thld_for_valid"]).float()
                epoch_valid_metric += m_batch_metric.item()
            epoch_valid_metric = epoch_valid_metric / (i_valid + 1)
            print("\n")
            print(
                "VALIDATION >>> epoch: {:04d}/{:04d}, running_metric: {}".format(
                    epoch + 1,
                    self.epochs,
                    epoch_valid_metric,
                ),
                end="\r",
            )
            print("\n" * 2)
        return epoch_valid_metric
