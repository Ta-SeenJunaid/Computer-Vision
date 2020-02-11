#Write a small application to find the Canny edge detection whose threshold values can be varied using two trackbars

import cv2

x = int(input('Enter  minVal: '))
y = int(input('Enter maxVal: '))

img = cv2.imread('../DATA/chessboard_mat.jpg',0)

cv2.imshow('Image Gray', img)

canny = cv2.Canny(img, x, y)
cv2.imshow('Canny', canny)
cv2.waitKey(0)

cv2.destroyAllWindows()