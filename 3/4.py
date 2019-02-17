import cv2
import numpy as np

import matplotlib.pyplot as plt

blank_img = np.zeros(shape=(512,512,3),dtype=np.int16)
blank_img.shape

plt.imshow(blank_img)

cv2.rectangle(blank_img,pt1=(384,0),pt2=(510,150),color=(0,255,0),thickness=10)

cv2.rectangle(blank_img,pt1=(200,200),pt2=(300,300),color=(0,0,255),thickness=10)

cv2.circle(img=blank_img,center=(100,100),radius=50,color=(255,0,0),thickness=8)

cv2.circle(img=blank_img,center=(400,400),radius=50,color=(255,0,0),thickness=-1)

cv2.line(blank_img,pt1=(0,0),pt2=(512,512),color=(102,255,255),thickness=5)

plt.imshow(blank_img)
