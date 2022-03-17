import os
import cv2
import numpy as np

path = '../IMG/test_boy.jpg'

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.resize(img, (48, 48))



cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
