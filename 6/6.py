# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('DATA/flat_chessboard.png')
#flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)

plt.imshow(flat_chess)

found,corners = cv2.findChessboardCorners(flat_chess,(7,7))

cv2.drawChessboardCorners(flat_chess,(7,7),corners,found)


dots = cv2.imread('DATA/dot_grid.png')
plt.imshow(dots)

found,corners = cv2.findCirclesGrid(dots,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(dots,(10,10),corners,found)
plt.imshow(dots)




