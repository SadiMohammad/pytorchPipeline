
from torch import optim
import torch
from tqdm import tqdm
from configparser import ConfigParser
from losses import *
from dataLoader import *
from utils import *
from losses import *
from eval import *
import sys,os
sys.path.append('..')
from models import UNet


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

    def main(self, model, device):
        imgRawTrain = DataLoad(self.trainImagePath, self.imgRows, self.imgCols).loadPathData()
        imgMaskTrain = DataLoad(self.trainMaskPath, self.imgRows, self.imgCols, False).loadPathData()
        imgRawTrainMeaned = DataLoad(self.trainImagePath, self.imgRows, self.imgCols).stdMeaned(imgRawTrain)
        imgMaskTrainNormed = DataLoad(self.trainMaskPath, self.imgRows, self.imgCols, False).normalized(imgMaskTrain)

        imgTrain, imgVal, maskTrain, maskVal = split_train_val(imgRawTrainMeaned, imgMaskTrainNormed, 0.1)
        modelName = model.__class__.__name__

        print('''
            Starting training:
                Model name: {}
                Epochs: {}
                Batch size: {}
                Learning rate: {}
                Training size: {}
                Validation size: {}
                Checkpoints: {}
                DEVICE: {}
            '''.format(modelName, self.epochs, self.batchSize, self.learningRate, len(imgTrain),
                       len(imgVal), str(self.saveBestModel), str(device)))

        optimizer = optim.Adam(model.parameters(),
                              lr=self.learningRate,
                              weight_decay=0.0005)

        for epoch in tqdm(range(self.epochs)):
            print('Starting epoch {}/{}.'.format(epoch + 1, self.epochs))
            model.train()

            bestDiceCoeff = 0
            epochLoss = 0
            epochTrainLoss = 0
            trainZipped = zip(imgTrain, maskTrain)

            for i, b in enumerate(batch(trainZipped, self.batchSize)):
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                trueMasks = np.array([i[1] for i in b]).astype(np.float32)

                imgs = torch.from_numpy(imgs).float().to(device)
                trueMasks = torch.from_numpy(trueMasks).float().to(device)

                predMasks = model(imgs)
                predMasksFlat = predMasks.view(-1)

                trueMasksFlat = trueMasks.view(-1)

                loss = Loss(trueMasksFlat, predMasksFlat).dice_coeff_loss()
                epochLoss += loss
                trainDice = Loss(trueMasksFlat, predMasksFlat).dice_coeff()
                epochTrainLoss += trainDice

                print('{0:.4f} --- loss: {1:.6f}'.format(i * self.batchSize / len(imgTrain), loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Epoch finished ! Loss: {}'.format(epochLoss / i))
                print(' ! Train Dice Coeff: {}'.format(epochTrainLoss / i))

                valZipped = zip(imgVal, maskVal)
                valDice = evalModel(model, valZipped, device)
                print('Validation Dice Coeff: {}'.format(valDice))

                try:
                    # Create model Directory
                    os.mkdir(self.checkpointsPath + '/' + modelName)
                    print("Directory ", modelName, " Created ")
                except FileExistsError:
                    print("Directory ", modelName, " already exists")

                if self.saveBestModel & valDice>bestDiceCoeff:
                    bestDiceCoeff = valDice
                    torch.save(model.state_dict(),
                               self.checkpointsPath + '/' + modelName + '/' + 'CP_epoch-{}_valDice-{}.pth'.format((epoch + 1), valDice))
                    print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(3, 1).to(device)
    try:
        train().main(model, device)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), train().checkpointsPath + '/' + model.__class__.__name__ + '/' +'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
