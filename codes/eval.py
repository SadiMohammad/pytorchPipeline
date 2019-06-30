
import torch
from losses import *


def evalModel(model, zippedVal, device):
    model.eval()
    totValLoss = 0
    for i, b in enumerate(zippedVal):
        imgs = b[0]
        trueMasks = b[1]

        imgs = torch.from_numpy(imgs).float().to(device)
        trueMasks = torch.from_numpy(trueMasks).float().to(device)

        trueMasksFlat = trueMasks.view(-1)

        predMasks = model(imgs)
        predMasksFlat = predMasks.view(-1)
        predMasksFlat = (predMasksFlat > 0.5).float()

        totValLoss += Loss(trueMasksFlat, predMasksFlat).dice_coeff()
    return totValLoss / (i + 1)