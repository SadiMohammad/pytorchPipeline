import torch

__all__ = [
    "dice_coeff_loss",
    "iou_calc_loss",
    "bce_logit_loss",
    "bce_logit_with_dice_loss",
]


class Flatten:
    def forward(self, input):
        return input.view(input.size(0), -1)


class Loss:
    def __init__(self, y_true, y_pred):
        self.smooth = 1e-7
        self.y_true_f = Flatten().forward(y_true)
        self.y_pred_f = Flatten().forward(y_pred)
        self.intersection = torch.sum(self.y_true_f * self.y_pred_f, dim=1)
        self.union = (
            torch.sum(self.y_true_f) + torch.sum(self.y_pred_f)
        ) - self.intersection

    def dice_coeff_loss(self):
        loss = (2.0 * self.intersection + self.smooth) / (
            torch.sum(self.y_true_f, dim=1)
            + torch.sum(self.y_pred_f, dim=1)
            + self.smooth
        )
        # loss = loss.type(torch.DoubleTensor)
        return -loss

    def iou_calc_loss(self):
        return -((self.intersection + self.smooth) / (self.union + self.smooth))

    def bce_logit_loss(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(self.y_pred_f, self.y_true_f)
        return loss

    def bce_logit_with_dice_loss(self):
        bce_loss = self.bce_logit_loss()
        dice_loss = self.dice_coeff_loss()
        loss = bce_loss + dice_loss
        return loss
