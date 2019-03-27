# -*- coding: utf-8 -*-

import cv2

img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt
plt.imshow(img1)

plt.imshow(img2)