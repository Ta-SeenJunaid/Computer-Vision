
import cv2

face_cascade = cv2.CascadeClassifier('../data2/haar-cascade-files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data2/haar-cascade-files/haarcascade_eye.xml')

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        