
import torch
from losses import *
import numpy as np
from utils import *

def evalModel(model, zippedVal, device):
    totValDice = 0
    for i, b in enumerate(batch(zippedVal)):
        imgs = np.array(b[0]).astype(np.float32)
        imgs = imgs.reshape(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
        trueMasks = np.array(b[1]).astype(np.float32)

        imgs = torch.from_numpy(imgs).float().to(device)
        trueMasks = torch.from_numpy(trueMasks).float().to(device)

        predMasks = model(imgs)
        predMasks = (predMasks > 0.5).float()

        valDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
        print(valDice)
        totValDice += valDice.item()
    return totValDice / (i + 1)