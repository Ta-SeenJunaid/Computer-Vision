# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DATA/internal_external.png',0)
plt.imshow(img,cmap='gray')

image,contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(image.shape)



