import copy
import datetime
import glob
import os
import sys
import torchvision.transforms as transforms
from configparser import ConfigParser
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils import *

sys.path.append('..')
from models import SegNet
# from models import CleanU_Net
# from modeling import DeepLab

class config:
    def __init__(self):
        # parser config
        config_file = "./config.ini"
        parser = ConfigParser()
        parser.read(config_file)

        #default config
        self.checkpointsPath = parser["DEFAULT"].get("checkpointsPath")

        # test config
        self.testImagePath = parser["TEST"].get("testImagePath")
        self.testMaskPath = parser["TEST"].get("testMaskPath")
        self.predMaskPath = parser["TEST"].get("predMaskPath")
        self.size = parser["TEST"].getint("size")
        self.batchSize = parser["TEST"].getint("batchSize")
        self.modelWeight = parser["TEST"].get("modelWeight")

class test(config):
    def __init__(self):
        super().__init__()

    def main(self, model, device):
        modelName = model.__class__.__name__
        imageFiles = glob.glob(self.testImagePath + "/*" + ".jpg")
        print(imageFiles)
        maskFiles = glob.glob(self.testMaskPath + "/*" + ".tif")
        print(maskFiles)

        datasetTest = Dataset_RAM(imageFiles, maskFiles, self.size)
        loaderTest = torch.utils.data.DataLoader(datasetTest, batch_size=self.batchSize, shuffle=True)

        for i_test, sample_test in enumerate(tqdm(loaderTest)):
            model.eval()
            images = sample_test[0].to(device)
            trueMasks = sample_test[1].to(device)
            predMasks = model(images)

            plt.figure()
            predTensor = (torch.exp(predMasks[0, 0, :, :]).detach().cpu())
            plt.imshow((predTensor/torch.max(predTensor))*255, cmap='gray')
            pilTrans = transforms.ToPILImage()
            pilImg = pilTrans((predTensor/torch.max(predTensor))*255)
            pilArray = np.array(pilImg)
            pilArray = (pilArray > 127)
            im = Image.fromarray(pilArray)
            im.save(self.predMaskPath + '/' + str(i_test) + '.tif')

            print((predTensor/torch.max(predTensor))*255)

            mBatchDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
            print(mBatchDice.item())

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SegNet(1, 1).to(device)
    # model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
    modelName = model.__class__.__name__
    checkpoint = torch.load(test().checkpointsPath + '/' + modelName + '/' + test().modelWeight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    test().main(model, device)
