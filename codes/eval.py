
import torch
from losses import *
import numpy as np

def evalModel(model, zippedVal, device):
    model.eval()
    totValLoss = 0
    for i, b in enumerate(zippedVal):
        imgs = np.array(b[0]).astype(np.float32)
        imgs = imgs.reshape(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
        trueMasks = np.array(b[1]).astype(np.float32)

        imgs = torch.from_numpy(imgs).float().to(device)
        trueMasks = torch.from_numpy(trueMasks).float().to(device)

        predMasks = model(imgs)
        predMasks = (predMasks > 0.5).float()

        totValLoss += Loss(trueMasks, predMasks).dice_coeff()
    return totValLoss / (i + 1)