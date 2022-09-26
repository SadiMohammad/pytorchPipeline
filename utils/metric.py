import torch

__all__ = [
    "dice_coeff",
    "iou_calc",
]


class Flatten:
    def forward(self, input):
        return input.view(input.size(0), -1)


class Metric:
    def __init__(self, y_true, y_pred):
        self.smooth = 1e-7
        self.y_true_f = Flatten().forward(y_true)
        self.y_pred_f = Flatten().forward(y_pred)
        self.intersection = torch.sum(self.y_true_f * self.y_pred_f, dim=1)
        self.union = (
            torch.sum(self.y_true_f) + torch.sum(self.y_pred_f)
        ) - self.intersection

    def dice_coeff(self):
        coeff = (2.0 * self.intersection + self.smooth) / (
            torch.sum(self.y_true_f, dim=1)
            + torch.sum(self.y_pred_f, dim=1)
            + self.smooth
        )
        # coeff = coeff.type(torch.DoubleTensor)
        return coeff

    def iou_calc(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)
