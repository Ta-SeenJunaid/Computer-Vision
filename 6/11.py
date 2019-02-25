# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    
img = cv2.imread('DATA/pennies.jpg')
#display(img)

img = cv2.medianBlur(sep_coins,35)
display(img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#display(gray)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2) 
display(opening)

sure_bg = cv2.dilate(opening,kernel,iterations=3)
display(sure_bg)
 
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
display(dist_transform)

ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
display(sure_fg)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
display(unknown)

ret , markers = cv2.connectedComponents(sure_fg)

markers = markers +1
markers[]


