import configparser
import sys
sys.path.append('..')
from models import UNet

def getConfig():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    trainImagePath = cp["DEFAULT"].get("trainImagePath")
    trainMaskPath = cp["DEFAULT"].get("trainMaskPath")
    checkpointsPath = cp["DEFAULT"].get("checkpointsPath")

    # train config
    learningRate = cp["TRAIN"].getboolean("learningRate")
    optimizer = cp["TRAIN"].getboolean("optimizer")
    loss = cp["TRAIN"].getboolean("loss")
    imgRows = cp["TRAIN"].get("imgRows")
    imgCols = cp["TRAIN"].getint("imgCols")
    epochs = cp["TRAIN"].getint("epochs")
    batchSize = cp["TRAIN"].getfloat("batchSize")
    modelWeightLoad = cp["TRAIN"].getint("modelWeightLoad")