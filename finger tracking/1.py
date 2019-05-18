import cv2
import numpy as np

from sklearn.metrics import pairwise

background = None

accumalated_weight = 0.5

roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


def calc_accum_avg(frame,accumalated_weight):
    
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)
    
