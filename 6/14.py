import cv2
import numpy as np
import matplotlib.pyplot as plt

nadia = cv2.imread('DATA/Nadia_Murad.jpg',0)
denis = cv2.imread('DATA/Denis_Mukwege.jpg',0)
solvay = cv2.imread('DATA/solvay_conference.jpg',0)

face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_alt.xml')

def detect_face(img):
    
    face_img = img.copy()
    
    face_rects = face_cascade.detectMultiScale(face_img)
     
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        
    return face_img

result = detect_face(denis)
plt.imshow(result,cmap='gray')

result = detect_face(nadia)
plt.imshow(result,cmap='gray')
    
result = detect_face(solvay)
plt.imshow(result,cmap='gray')    
    

def adj_detect_face(img):
    
    face_img = img.copy()
    
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
     
    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        
    return face_img 

result = adj_detect_face(solvay)
plt.imshow(result,cmap='gray') 
    
eye_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    
    face_img = img.copy()
    
    eyes_rects = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
     
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
        
    return face_img

result = detect_eyes(nadia)
plt.imshow(result,cmap='gray')

