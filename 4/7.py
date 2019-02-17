# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DATA/bricks.jpg').astype(np.float32)/255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gamma = 2
result = np.power(img,gamma)
plt.imshow(result)
wt = img
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(wt,text='bricks', org=(10,600),fontFace=font,fontScale=10,
            color=(255,0,0),thickness=4)
plt.imshow(wt)
kernel = np.ones(shape=(5,5),dtype=np.float32)/25

dst = cv2.filter2D(wt,-1,kernel)
plt.imshow(dst)

blurred = cv2.blur(wt,ksize=(15,15))
plt.imshow(blurred)

gblurred = cv2.GaussianBlur(wt,(5,5),10)
plt.imshow(gblurred)

median_blur = cv2.medianBlur(wt,5)
plt.imshow(median_blur)

biblur = cv2.bilateralFilter(wt,9,75,75)
plt.imshow(biblur)
imgd = cv2.imread('DATA/sammy.jpg')
imgd = cv2.cvtColor(imgd,cv2.COLOR_BGR2RGB)

noise_img = cv2.imread('DATA/sammy_noise.jpg')
noise_img = cv2.cvtColor(noise_img,cv2.COLOR_BGR2RGB)
plt.imshow(noise_img)

median = cv2.medianBlur(noise_img,7)
plt.imshow(median)




