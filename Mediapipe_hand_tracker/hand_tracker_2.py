import cv2
import time
import hand_detection_and_tracking as hdt

w_cam, h_cam = 1280, 1280
cap = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1500, 1000)
p_time = 0

detector = hdt.HandDetectionAndTracking()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (30, 40), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)