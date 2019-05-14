

import torch

#https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2
class Flatten:
    def forward(self, input):
        return input.view(input.size(0), -1)

def iou_calc(y_true, y_pred):
    ones = torch.ones_like(y_true)
    y_true_inv = ones - y_true
    y_pred_inv = ones - y_pred
    tp = torch.sum(y_true * y_pred, dim = 1)
    fp = torch.sum(y_true_inv * y_pred, dim = 1)
    fn = torch.sum(y_true * y_pred_inv, dim = 1)

    iou = torch.mean(tp / (tp + fp + fn + safe))

    return iou

class Loss:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.smooth = 1

    def dice_coeff(self):
        y_true_f = Flatten.forward(self.y_true)
        y_pred_f = Flatten.forward(self.y_pred)
        intersection = torch.sum(y_true_f * y_pred_f, dim = 1)
        return (2. * intersection + smooth) / (torch.sum(y_true_f, dim = 1) + torch.sum(y_pred_f, dim = 1) + self.smooth)
