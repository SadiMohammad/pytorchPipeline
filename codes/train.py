import copy
import argparse
import datetime
import glob
import os
import sys
from configparser import ConfigParser

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision

from utils import *
from losses import *

sys.path.append('..')
# from models import UNet
# from models import CleanU_Net
from models import LungNet


class config:
    def __init__(self, scheme):
        # parser config
        self.scheme = scheme
        print(scheme)
        print(type(scheme))
        config_file = "./config.ini"
        parser = ConfigParser()
        parser.read(config_file)

        # config
        self.imagesPath = parser[self.scheme].get("imagesPath")
        self.masksPath = parser[self.scheme].get("masksPath")
        self.checkpointsPath = parser[self.scheme].get("checkpointsPath")
        self.learningRate = parser[self.scheme].getfloat("learningRate")
        self.trainRatio = parser[self.scheme].getfloat("trainRatio")
        self.optimizer = parser[self.scheme].get("optimizer")
        self.loss = parser[self.scheme].get("loss")
        self.size = parser[self.scheme].getint("size")
        self.epochs = parser[self.scheme].getint("epochs")
        self.batchSize = parser[self.scheme].getint("batchSize")
        self.modelWeightLoad = parser[self.scheme].getboolean("modelWeightLoad")
        self.modelWeight = parser[self.scheme].get("modelWeight")
        self.saveBestModel = parser[self.scheme].getboolean("saveBestModel")


class train(config):
    def __init__(self, scheme):
        super().__init__(scheme)

    def main(self, model, device):
        modelName = model.__class__.__name__
        imageFiles = glob.glob(self.imagesPath + "/*" + ".jpg")
        maskFiles = glob.glob(self.masksPath + "/*" + ".tif")

        imgTrain = imageFiles[:int(len(imageFiles) * self.trainRatio)]
        imgVal = imageFiles[int(len(imageFiles) * self.trainRatio):]
        maskTrain = maskFiles[:int(len(imageFiles) * self.trainRatio)]
        maskVal = maskFiles[int(len(imageFiles) * self.trainRatio):]

        print('''
            Starting training:
                Model name: {}
                Epochs: {}
                Batch size: {}
                Learning rate: {}
                Total size: {}
                Training size: {}
                Validation size: {}
                Checkpoints: {}
                DEVICE: {}
            '''.format(modelName, self.epochs, self.batchSize, self.learningRate, len(imageFiles), len(imgTrain),
                       len(imgVal), str(self.saveBestModel), str(device)))

        # # params = [p for p in model_ft.parameters() if p.requires_grad]
        # optimizer = optim.SGD(model.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=0.00005)

        optimizer = optim.Adam(model.parameters(),
                               lr=self.learningRate,
                               weight_decay=0.0005)

        datasetTrain = Dataset_ROM(imgTrain, maskTrain, self.size, convert='L')
        loaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=self.batchSize, shuffle=True)

        datasetValid = Dataset_ROM(imgVal, maskVal, self.size, convert='L')
        loaderValid = torch.utils.data.DataLoader(datasetValid, batch_size=self.batchSize, shuffle=True)

        bestDiceCoeff = 0.4607001204974949

        for epoch in range(self.epochs):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            model.train()

            epochWorstDiceCoeff = 1
            epochTrainLoss = 0
            epochTrainDice = 0
            epochValDice = 0
            for i_train, sample_train in enumerate(tqdm(loaderTrain)):
                images = sample_train[0].to(device)
                trueMasks = sample_train[1].to(device)
                preds = model(images)
                predMasks = preds

                # for deeplabv3
                # preds = preds['out'] 
                # predMasks = torch.sigmoid(preds) 
                # print(predMasks.values())

                mBatchLoss = torch.mean(Loss(trueMasks, predMasks).dice_coeff_loss())
                epochTrainLoss += mBatchLoss.item()
                mBatchDice = torch.mean(Loss(trueMasks, predMasks).dice_coeff())
                epochTrainDice += mBatchDice.item()

                optimizer.zero_grad()
                mBatchLoss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    mBatchValDice = evalModel(model, loaderValid, device)
                    epochValDice += mBatchValDice

                saveCheckpoint = {'epoch': epoch,
                                  'input_size': self.size,
                                  'best_dice': bestDiceCoeff,
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'model_state_dict': model.state_dict()}

                if mBatchValDice > bestDiceCoeff:
                    bestDiceCoeff = mBatchValDice
                    torch.save(saveCheckpoint,
                               self.checkpointsPath + '/' + modelName + '/' + '{}_epoch-{}_dice-{}.pth'.format(
                                   str(datetime.datetime.now()),
                                   (epoch + 1), bestDiceCoeff))
                    print('Checkpoint {} saved !'.format(epoch + 1))
                    # best_model = copy.deepcopy(model)
                if mBatchValDice < epochWorstDiceCoeff:
                    epochWorstDiceCoeff = mBatchValDice

            print('Epoch finished ! Epoch Train Dice Coeff: {}'.format(epochTrainDice / (i_train + 1)))
            print(' ! Epoch Worst Validation Dice Coeff: {}'.format(epochWorstDiceCoeff))
            print(' ! Epoch Validation Dice Coeff: {}'.format(epochValDice / (i_train + 1)))
            print(' ! Best Validation Dice Coeff: {}'.format(bestDiceCoeff))


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config_scheme', help='configuration for train.py', required=True)
    args = argParser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LungNet(1, 1).to(device)
    # checkpoint = torch.load(train(args.config_scheme).checkpointsPath + '/' + modelName + '/' + '2019-08-01 04:39:48.331969_epoch-10_dice-0.7213704585097731.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# epoch = checkpoint['epoch']
	# loss = checkpoint['loss']
    # model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1)
    modelName = model.__class__.__name__
    model = model.to(device)
    checkpoint = torch.load(train(args.config_scheme).checkpointsPath + '/' + modelName + '/' + train(args.config_scheme).modelWeight)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        # Create model Directory
        checkpointDir = train(args.config_scheme).checkpointsPath + '/' + modelName
        if not (os.path.exists(checkpointDir)):
            os.mkdir(checkpointDir)
            print("\nDirectory ", modelName, " Created \n")
        train(args.config_scheme).main(model, device)
    except KeyboardInterrupt:
        torch.save(model.state_dict(),
                   train(args.config_scheme).checkpointsPath + '/' + model.__class__.__name__ + '/' + 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
