

import numpy as np
import cv2
import glob

class DataLoad:
    def __init__(self, dirImage, img_rows, img_cols):
        self.dirImage = dirImage
        self.img_rows = img_rows
        self.img_cols = image_cols

    def loadPathData(self):
        filenames= []
        filenames += glob.glob(dirImage + "/*" + ".jpg")
        images = []
        for file in filenames:
            img = cv2.imread(file,0)
            img= cv2.resize(img, (img_rows, img_cols), interpolation = cv2.INTER_AREA)
            img = np.asarray(img, dtype = np.bool)
            images.append(img)
        images = np.asarray(images, dtype = np.bool)
        print(np.shape(images))
        images = images.reshape(-1, img_rows, img_cols, 1)
        print(np.shape(images))
        return images
