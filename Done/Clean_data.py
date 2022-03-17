import os
import cv2
import numpy as np

errors = list()
datafile = '../Data/'
Tr = list()

if __name__ == '__main__':

    for file in os.listdir(datafile):
        img_path = datafile + file
        for path in os.listdir(img_path):
            path = img_path+'/'+path

            img = cv2.imread(path)
            img = np.array(img)
            if img.shape[0] != 48 or img.shape[1] != 48:
                errors.append(path)
            else:
                Tr.append(path)

    print(len(Tr))