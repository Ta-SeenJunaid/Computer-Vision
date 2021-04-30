import cv2
import time
import hand_detection_and_tracking as hdt
import math

cap = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1600, 1000)
nm_list = [0]
p_time = 0

detector = hdt.HandDetectionAndTracking()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    img = cv2.rectangle(img, (0,0), (1600,1600), (255, 255, 255), -1)
    img = cv2.rectangle(img, (160,50), (255, 100), (40,150,255), -1)
    cv2.putText(img, '8', (200, 85), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 0), 2)
    img = cv2.rectangle(img, (160,200), (255, 250), (40,150,255), -1)
    cv2.putText(img, '0', (200, 230), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 0), 2)


    lm_list = detector.find_position(img)

    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])

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
            if cx in range(160, 250) and cy in range(50, 100):
                cv2.rectangle(img, (155, 45), (260, 105), (0, 255, 0), 2)
                nm_list[0] = 8
            if cx in range(160, 255) and cy in range(200, 250):
                cv2.rectangle(img, (155, 195), (260, 255), (0, 255, 0), 2)
                nm_list[0] = 0


    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f'FPS: {int(fps)}', (30, 40), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 2)
    cv2.putText(img, str(nm_list[0]), (200, 185), cv2.FONT_HERSHEY_COMPLEX,
                2, (0, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(4) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()