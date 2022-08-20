from models.unet import UNet
import os
import torch, torchvision
from utils.config import Cfgs, Config


class Model(Config):
    def __init__(self, args):
        super().__init__(args)

    def model_call(self):
        if self.cfgs["model"]["model_name"].lower() == "unet":
            return UNet(
                self.cfgs["model"]["in_channel"], self.cfgs["model"]["n_classes"]
            )
        if self.cfgs["model"]["model_name"].lower() == "deeplabv3":
            return torchvision.models.segmentation.deeplabv3_resnet101(
                pretrained=self.cfgs["model"]["pretrained"],
                num_classes=self.cfgs["model"]["n_classes"],
            )

    def load_weights(self, model, optimizer):
        checkpoint = torch.load(
            os.path.join(
                "../ckpts",
                self.cfgs["model"]["model_name"].lower(),
                self.cfgs["train_setup"]["model_weight_path"],
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
