
import torch
from losses import *
import numpy as np
from utils import *

def evalModel(model, validDataset):
    totValDice = 0
    for i_valid, sample_valid in enumerate(validDataset):
        images = sample_valid[0]
        trueMasks = sample_valid[1]

        predMasks = model(images)
        predMasks = (predMasks > 0.5).float()

        valDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
        print(valDice)
        totValDice += valDice.item()
    return totValDice / (i_valid + 1)