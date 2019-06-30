from sklearn.utils import shuffle
import random
from sklearn.model_selection import train_test_split
import os
from dataLoader import *

# def get_ids(dir):
#     """Returns a list of the ids in the directory"""
#     return list(f[:-4] for f in os.listdir(dir))
#
#
# def split_ids(ids, n=1):
#     """Split each id in n, creating n tuples (id, k) for each id"""
#     return list((id, i) for id in ids for i in range(n))

# def split_train_val(dataset, val_percent=0.05):
#     dataset = list(dataset)
#     length = len(dataset)
#     n = int(length * val_percent)
#     shuffle(dataset, random_state=0)
#     return {'train': dataset[:-n], 'val': dataset[-n:]}

# def batch(imgRaw, imgMask, batchSize):
#     batchImgRaw = []
#     batchImgMask = []

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

        if len(b) > 0:
            yield b

#    randomState = random.choice(np.arange(10)) #changes in epoch
#    imgRaw = shuffle(imgRaw, random_state=randomState)
#    imgMask = shuffle(imgMask, random_state=randomState)
# def raw_mask_batch(imgRaw, imgMask, batchSize, batchIter):
#




def split_train_val(X, y, val_size = 0.1, random_state = 0):
    return train_test_split(X, y, test_size = val_size, random_state = random_state)

if __name__ == "__main__":
    # dir = '../data/train/raw all'
    # ids = get_ids(dir)
    # # ids = split_ids(ids)
    # val_percent = 0.1
    # iddataset = split_train_val(ids, val_percent)
    imgRawTrain = DataLoad('../data/train/raw all', 128, 128).loadPathData()
    imgMaskTrain = DataLoad('../data/train/mask all', 128, 128, False).loadPathData()
    imgRawTrainMeaned = DataLoad('../data/train/raw all', 128, 128).stdMeaned(imgRawTrain)
    imgMaskTrainNormed = DataLoad('../data/train/mask all', 128, 128, False).normalized(imgMaskTrain)
    imgTrain, imgVal, maskTrain, maskVal = train_test_split(imgRawTrainMeaned, imgMaskTrainNormed, test_size = 0.1, random_state = 0)
