import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/crossword.jpg',0)

#plt.imshow(img,cmap='gray')

ret,th1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
#plt.imshow(th1,cmap='gray')

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
#plt.imshow(th2,cmap='gray')

blended = cv2.addWeighted(src1=th1,alpha=0.5,src2=th2,beta=0.5,gamma=0)
plt.imshow(th2,cmap='gray')