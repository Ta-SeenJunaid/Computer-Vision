# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DATA/sudoku.jpg',0)
plt.imshow(img,cmap='gray')

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
plt.imshow(soblex,cmap='gray')

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.imshow(sobley,cmap='gray')

#sobley_x_y = cv2.Sobel(img,cv2.CV_64F,0,0,ksize=5)
#sobley_x_y=soblex+sobley
#plt.imshow(sobley_x_y,cmap='gray')

laplacian = cv2.Laplacian(img,cv2.CV_64F)
plt.imshow(laplacian,cmap='gray')

blended = cv2.addWeighted(src1=sobelx, alpha=0.5,src2=sobely,beta=0.5,gamma=0)
plt.imshow(blended,cmap='gray')

ret, th1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
plt.imshow(th1,cmap='gray')

kernel = np.ones((4,4),np.uint8)
gradient = cv2.morphologyEx(blended,cv2.MORPH_GRADIENT,kernel)
plt.imshow(gradient,cmap='gray')
