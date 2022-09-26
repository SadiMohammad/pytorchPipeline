from models.unet import UNet
import os
import torch, torchvision


class Model:
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def get_model(self):
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
                self.cfgs["train_setup"]["checkpoints_path"],
                self.cfgs["model"]["model_name"].lower(),
                self.cfgs["train_setup"]["model_weight_path"],
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
