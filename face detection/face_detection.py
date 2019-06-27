
import cv2

face_cascade = cv2.CascadeClassifier('../data2/haar-cascade-files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data2/haar-cascade-files/haarcascade_eye.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)