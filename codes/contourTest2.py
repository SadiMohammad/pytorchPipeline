import os
import logging

import copy
import numpy as np
import datetime
import glob
import os, logging
import sys
from configparser import ConfigParser

from skimage import filters
from skimage.measure import regionprops
import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

import morphsnakes as ms
from utils import *
from losses import *

sys.path.append('..')
from models import SegNet, UNet
# from models import CleanU_Net
# from modeling import DeepLab



class config:
    def __init__(self):
        # parser config
        config_file = "./config.ini"
        parser = ConfigParser()
        parser.read(config_file)

        # default config
        self.trainImagePath = parser["DEFAULT"].get("trainImagePath")
        self.trainMaskPath = parser["DEFAULT"].get("trainMaskPath")
        self.checkpointsPath = parser["DEFAULT"].get("checkpointsPath")

        # train config
        self.learningRate = parser["TRAIN"].getfloat("learningRate")
        self.trainRatio = parser["TRAIN"].getfloat("trainRatio")
        self.optimizer = parser["TRAIN"].get("optimizer")
        self.loss = parser["TRAIN"].get("loss")
        self.size = parser["TRAIN"].getint("size")
        self.epochs = parser["TRAIN"].getint("epochs")
        self.batchSize = parser["TRAIN"].getint("batchSize")
        self.modelWeightLoad = parser["TRAIN"].getboolean("modelWeightLoad")
        self.modelWeight = parser["TRAIN"].get("modelWeight")
        self.saveBestModel = parser["TRAIN"].getboolean("saveBestModel")

        # test config
        self.testImagePath = parser["TEST"].get("testImagePath")
        self.testMaskPath = parser["TEST"].get("testMaskPath")
        self.predMaskPath = parser["TEST"].get("predMaskPath")
        self.size = parser["TEST"].getint("size")
        self.batchSize = parser["TEST"].getint("batchSize")
        self.modelWeight = parser["TEST"].get("modelWeight")

class morphSnake:
    def __init__(self, maskPred, center, dia, iterations):
        self.maskPred = maskPred
        self.center = center
        self.dia = dia
        self.iterations = iterations

    def visual_callback_2d(self, background, fig=None):
        # Prepare the visual environment.
        if fig is None:
            fig = plt.figure()
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(background, cmap=plt.cm.gray)

        ax2 = fig.add_subplot(1, 2, 2)
        ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
        plt.pause(0.001)

        def callback(levelset):

            if ax1.collections:
                del ax1.collections[0]
            ax1.contour(levelset, [0.5], colors='r')
            ax_u.set_data(levelset)
            fig.canvas.draw()
            plt.pause(0.001)

        return callback

    def rgb2gray(self, img):
        """Convert a RGB image to gray scale."""
        return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    def example_lakes(self):
        logging.info('Running: example_lakes (MorphACWE)...')

        # Load the image.
        img = self.maskPred

        # MorphACWE does not need g(I)

        # Initialization of the level-set.
        init_ls = ms.circle_level_set(img.shape, (self.center[0], self.center[1]), (self.dia) / 2)

        # Callback for visual plotting
        callback = self.visual_callback_2d(img)

        # Morphological Chan-Vese (or ACWE)
        ms.morphological_chan_vese(img, iterations=self.iterations,
                                   init_level_set=init_ls,
                                   smoothing=3, lambda1=1, lambda2=1,
                                   iter_callback=callback)


class train(config):
    def __init__(self):
        super().__init__()

    def main(self, model1, model2, model3, device):
        modelName1 = model1.__class__.__name__
        modelName2 = model2.__class__.__name__
        modelName3 = model3.__class__.__name__
        imageFiles = glob.glob(self.testImagePath + "/*" + ".jpg")
        print(imageFiles)
        maskFiles = glob.glob(self.testMaskPath + "/*" + ".tif")
        print(maskFiles)

        datasetTest13 = Dataset_RAM_TEST(imageFiles, self.size)
        loaderTest13 = torch.utils.data.DataLoader(datasetTest13, batch_size=self.batchSize, shuffle=False)

        datasetTest2 = Dataset_RAM_TEST(imageFiles, self.size, convert='RGB')
        loaderTest2 = torch.utils.data.DataLoader(datasetTest2, batch_size=self.batchSize, shuffle=False)

        model1.eval()
        model2.eval()
        model3.eval()
        i = 0
        for sample_test in zip(loaderTest13, loaderTest2):
            print(i)
            i += 1
            images13 = sample_test[0].to(device)
            images2 = sample_test[1].to(device)
            preds1 = model1(images13)
            preds2 = model2(images2)
            preds3 = model3(images13)
            predMasks1 = preds1
            predMasks2 = torch.sigmoid(preds2['out'])
            predMasks3 = preds3
            predArray1 = predMasks1.detach().cpu().numpy()
            predArray2 = predMasks2.detach().cpu().numpy()
            predArray3 = predMasks3.detach().cpu().numpy()
            imagesArray = images13.detach().cpu().numpy()
            predArray1 = np.where(predArray1 > 0.5, 1, 0)
            predArray2 = np.where(predArray2 > 0.5, 1, 0)
            predArray3 = np.where(predArray3 > 0.5, 1, 0)
            for b in range(predArray1.shape[0]):
                predArray1 = predArray1[b, 0, :, :]
                print(predArray1.shape)
                predArray2 = predArray2[b, 0, :, :]
                print(predArray2.shape)
                predArray3 = predArray3[b, 0, :, :]
                print(predArray3.shape)
                imagesArray = imagesArray[b, 0, :, :]
                threshold_value = filters.threshold_otsu(predArray1)
                labeled_foreground = (predArray1 > threshold_value).astype(int)
                properties = regionprops(labeled_foreground, predArray1)
                center_of_mass1 = properties[0].centroid
                dia1 = properties[0].equivalent_diameter
                threshold_value = filters.threshold_otsu(predArray2)
                labeled_foreground = (predArray2 > threshold_value).astype(int)
                properties = regionprops(labeled_foreground, predArray2)
                center_of_mass2 = properties[0].centroid
                dia2 = properties[0].equivalent_diameter
                threshold_value = filters.threshold_otsu(predArray3)
                labeled_foreground = (predArray3 > threshold_value).astype(int)
                properties = regionprops(labeled_foreground, predArray3)
                center_of_mass3 = properties[0].centroid
                dia3 = properties[0].equivalent_diameter
                fig, ax = plt.subplots()
                ax.imshow(imagesArray, cmap='gray')
                ax.scatter(center_of_mass1[1], center_of_mass1[0], s=160, c='green', marker='+', label='SegNet')
                ax.scatter(center_of_mass2[1], center_of_mass2[0], s=160, c='blue', marker='s', label='DeeplabV3')
                ax.scatter(center_of_mass3[1], center_of_mass3[0], s=160, c='red', marker='o', label='UNet')
                plt.show()

                # logging.basicConfig(level=logging.DEBUG)
                # morphSnake(imagesArray, center_of_mass, dia/2, 200).example_lakes()
                # logging.info("Done.")
                # plt.show()

            # if i_train > 0:
            #     break
            #
            # break

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = SegNet(1, 1).to(device)
    modelName1 = model1.__class__.__name__
    model2 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
    model2 = model2.to(device)
    modelName2 = model2.__class__.__name__
    model3 = UNet(1, 1).to(device)
    modelName3 = model3.__class__.__name__

    model1_checkpoint = torch.load(train().checkpointsPath + '/' + modelName1 + '/'
                                   + '2019-08-30 13:21:52.559302_epoch-5_dice-0.4743926368317377.pth')
    model2_checkpoint = torch.load(train().checkpointsPath + '/' + modelName2 + '/'
                                   + '2019-08-22 08:37:06.839794_epoch-1_dice-0.4479589270841744.pth')
    model3_checkpoint = torch.load(train().checkpointsPath + '/' + modelName3 + '/'
                                   + '2019-09-03 03:21:05.647040_epoch-253_dice-0.46157537277322264.pth')

    model1.load_state_dict(model1_checkpoint['model_state_dict'])
    model2.load_state_dict(model2_checkpoint['model_state_dict'])
    model3.load_state_dict(model3_checkpoint['model_state_dict'])

    try:
        # Create model Directory
        train().main(model1, model2, model3, device)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
