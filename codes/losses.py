

import torch

# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2
class Flatten:
    def forward(self, input):
        return input.view(input.size(0), -1)

class Loss:
    def __init__(self, y_true, y_pred):
        self.smooth = 1
        self.y_true_f = Flatten.forward(y_true)
        self.y_pred_f = Flatten.forward(y_pred)
        self.intersection = torch.sum(self.y_true_f * self.y_pred_f, dim = 1)
        self.union = (torch.sum(self.y_true_f) + torch.sum(self.y_pred_f)) - self.intersection

    def dice_coeff(self):
        return (2. * self.intersection + self.smooth) / (torch.sum(self.y_true_f, dim = 1) + torch.sum(self.y_pred_f, dim = 1) + self.smooth)

    def iou_calc(self):
        return (self.intersection + self.smooth)/(self.union + self.smooth)
