

import numpy as np
import cv2
import glob

class DataLoad:
    def __init__(self, dirImage, img_rows, img_cols, dtype = np.uint8):
        self.dirImage = dirImage
        self.img_rows = img_rows
        self.img_cols = image_cols
        self.dtype = dtype

    def loadPathData(self):
        filenames= []
        filenames += glob.glob(dirImage + "/*" + ".jpg")
        images = []
        for file in filenames:
            img = cv2.imread(file,0)
            img= cv2.resize(img, (img_rows, img_cols), interpolation = cv2.INTER_AREA)
            img = np.asarray(img, dtype = self.dtype)
            images.append(img)
        images = np.asarray(images, dtype = self.dtype)
        print(np.shape(images))
        images = images.reshape(-1, 1, img_rows, img_cols)
        print(np.shape(images))
        return images

    def stdMeaned(self, batchImg):
        batchImg = batchImg.astype('float32')
        mean = np.mean(batchImg)
        std = np.std(batchImg)
        batchImg -= mean
        batchImg /= std
        batchImg /= 255.
        return batchImg
