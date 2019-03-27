# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/00-puppy.jpg')
#print(img)
#plt.imshow(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)