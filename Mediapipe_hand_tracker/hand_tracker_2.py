import cv2
import time
import hand_detection_and_tracking as hdt
import math

cap = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1600, 1000)
p_time = 0

detector = hdt.HandDetectionAndTracking()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=False)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        print(lm_list[4], lm_list[8])

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) //2

        cv2.circle(img, (x1, y1), 12, (255, 0, 0), 3)
        cv2.circle(img, (x1, y1), 3, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (0, 0, 255), 3)
        cv2.circle(img, (x2, y2), 3, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2 - y1)
        if length < 30:
            cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (30, 40), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)