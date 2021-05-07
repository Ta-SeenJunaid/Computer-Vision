import cv2
import numpy as np
import face_recognition
import os

path = 'image_attendance'
images = []
class_names = []
my_list = os.listdir(path)
print(my_list)
for cl in my_list:
    cur_img = cv2.imread(f'{path}/{cl}')
    images.append(cur_img)
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

encode_list_known = find_encodings(images)
print('Encoding Complete')