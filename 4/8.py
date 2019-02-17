# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

blank_img = np.zeros((600,600))
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(blank_img,text='ABCDE',org=(50,300),fontFace=font,fontScale=5,
            color=(255,255,255),thickness=30)
#plt.imshow(img,cmap='gray')

kernel = np.ones((5,5),dtype=np.uint8)
result = cv2.erode(img,kernel,iterations=4)
#plt.imshow(result,cmap='gray')

white_noise = np.random.randint(low=0,high=2,size=(600,600))
white_noise=white_noise * 255
#plt.imshow(white_noise,cmap='gray')

noise_img = white_noise + img
#plt.imshow(noise_img,cmap='gray')

opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)
plt.imshow(opening,cmap='gray')

black_noise = np.random.randint(low=0,high=2,size=(600,600))
black_noise = black_noise * -255

black_noise_img = img + black_noise
black_noise_img[black_noise_img == -255] = 0
#plt.imshow(black_noise_img,cmap='gray')

closing = cv2.morphologyEx(black_noise_img,cv2.MORPH_CLOSE,kernel)
plt.imshow(closing,cmap='gray')

gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
plt.imshow(gradient,cmap='gray')











