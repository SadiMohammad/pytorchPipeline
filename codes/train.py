
from torch import optim
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from configparser import ConfigParser
import sys
sys.path.append('..')
from models import UNet
from losses import Loss
from dataLoader import DataLoad

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.optimizer = parser["TRAIN"].get("optimizer")
        self.loss = parser["TRAIN"].get("loss")
        self.imgRows = parser["TRAIN"].getint("imgRows")
        self.imgCols = parser["TRAIN"].getint("imgCols")
        self.epochs = parser["TRAIN"].getint("epochs")
        self.batchSize = parser["TRAIN"].getint("batchSize")
        self.modelWeightLoad = parser["TRAIN"].getboolean("modelWeightLoad")
        self.saveBestModel = parser["TRAIN"].getboolean("saveBestModel")

class train(config):
    def __init__(self):
        super().__init__()

    def main(self, model):
        imgRawTrain = DataLoad('../data/train/raw all', 128, 128).loadPathData()
        imgMaskTrain = DataLoad('../data/train/mask all', 128, 128, False).loadPathData()
        imgRawTrainMeaned = DataLoad('../data/train/raw all', 128, 128).stdMeaned(imgRawTrain)
        imgMaskTrainNorm = DataLoad('../data/train/raw all', 128, 128, False).normalized(imgMaskTrain)

        tensorRawTrain = torch.from_numpy(imgRawTrainMeaned).float().to(device)
        tensorMaskTrain = torch.from_numpy(imgMaskTrainNorm).float().to(device)

        print('''
            Starting training:
                Epochs: {}
                Batch size: {}
                Learning rate: {}
                Training size: {}
                Validation size: {}
                Checkpoints: {}
                CUDA: {}
            '''.format(self.epochs, self.batchSize, self.learningRate, len(iddataset['train']),
                       len(iddataset['val']), str(self.saveBestModel), str(device)))

        optimizer = optim.SGD(model.parameters(),
                              lr=self.learningRate,
                              momentum=0.9,
                              weight_decay=0.0005)

        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            model.train()




# if __name__ == "__main__":
#     config()