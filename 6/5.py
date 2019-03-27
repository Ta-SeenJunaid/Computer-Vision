# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/sammy_face.jpg')
plt.imshow(img)

edges = cv2.Canny(image=img,threshold1=127,threshold2=127)
plt.imshow(edges)

med_val = np.median(img)

lower = int(max(0,0.7*med_val))

upper = int(min(255,1.3*med_val))


edges = cv2.Canny(image=img,threshold1=lower,threshold2=upper+100)
plt.imshow(edges)

blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image=blurred_img,threshold1=lower,threshold2=upper+50)
plt.imshow(edges)

