import cv2
import numpy as np
import face_recognition

img_elon = face_recognition.load_image_file('image_basic/elon_musk_1.jpg')
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file('image_basic/elon_musk_2.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

img_test_bill = face_recognition.load_image_file('image_basic/bill_gates.jpg')
img_test_bill = cv2.cvtColor(img_test_bill, cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(img_elon)[0]
encode_ellon = face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon, (face_loc[3], face_loc[0]),
              (face_loc[1], face_loc[2]), (0, 255, 0), 4)

face_loc_test = face_recognition.face_locations(img_test)[0]
encode_ellon_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (face_loc_test[3], face_loc_test[0]),
              (face_loc_test[1], face_loc_test[2]), (0, 255, 0), 4)

face_loc_test_bill = face_recognition.face_locations(img_test_bill)[0]
encode_bill_test = face_recognition.face_encodings(img_test_bill)[0]
cv2.rectangle(img_test_bill, (face_loc_test_bill[3], face_loc_test_bill[0]),
              (face_loc_test_bill[1], face_loc_test_bill[2]), (0, 255, 0), 4)

results = face_recognition.compare_faces([encode_ellon], encode_ellon_test)
face_dis = face_recognition.face_distance([encode_ellon], encode_ellon_test)
print(results, face_dis)
cv2.putText(img_test, f'{results} {round(face_dis[0], 2)}', (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

results_bill = face_recognition.compare_faces([encode_ellon], encode_bill_test)
face_dis_bill = face_recognition.face_distance([encode_ellon], encode_bill_test)
print(results_bill, face_dis_bill)
cv2.putText(img_test_bill, f'{results_bill} {round(face_dis_bill[0], 2)}', (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', img_elon)
cv2.imshow('Elon Musk Test', img_test)
cv2.imshow('Bill Gates Test', img_test_bill)
cv2.waitKey(0)