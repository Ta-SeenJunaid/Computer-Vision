# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

rainbow = cv2.imread('DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

img = rainbow

img.shape

mask = np.zeros(img.shape[:2],np.uint8)
plt.imshow(mask,cmap='gray')

mask[300:400,100:400] = 255
plt.imshow(mask,cmap='gray')

masked_img = cv2.bitwise_and(img,img,mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow,show_rainbow,mask=mask)
plt.imshow(show_masked_img)









