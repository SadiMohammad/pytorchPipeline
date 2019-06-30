

import numpy as np
import cv2
import glob

class DataLoad:
    def __init__(self, dirImage, img_rows = 128, img_cols = 128, raw = True):
        self.dirImage = dirImage
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.dtype = np.uint8
        self.data = raw

    def loadPathData(self):
        filenames= []
        if self.data:
            filenames += glob.glob(self.dirImage + "/*" + ".jpg")
        else:
            filenames += glob.glob(self.dirImage + "/*" + ".tif")
        images = []
        for file in filenames:
            img = cv2.imread(file, 0)
            img= cv2.resize(img, (self.img_rows, self.img_cols), interpolation = cv2.INTER_AREA)
            img = np.asarray(img, dtype = self.dtype)
            images.append(img)
        images = np.asarray(images, dtype = self.dtype)
        print(np.shape(images))
        images = images.reshape(-1, 1, self.img_rows, self.img_cols)
        print(np.shape(images))
        return images

    def stdMeaned(self, batchImg):
        batchImg = batchImg.astype('float32')
        batchImg /= 255.
        mean = np.mean(batchImg)
        std = np.std(batchImg)
        batchImg -= mean
        batchImg /= std
        return batchImg
    def normalized(self, batchImg):
        batchImg = (batchImg > 127).astype(np.uint)
        return batchImg

if __name__ == "__main__":
    imgRawTrain = DataLoad('../data/train/raw all', 128, 128).loadPathData()
    imgMaskTrain = DataLoad('../data/train/mask all', 128, 128, False).loadPathData()
    imgRawTrainMeaned = DataLoad('../data/train/raw all', 128, 128).stdMeaned(imgRawTrain)
    imgMaskTrainNormed = DataLoad('../data/train/mask all', 128, 128, False).normalized(imgMaskTrain)