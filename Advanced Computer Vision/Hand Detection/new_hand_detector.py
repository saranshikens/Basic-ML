import cv2 as cv
import mediapipe as mp
import time
import Hand_Tracking_Module as htm

prevTime = 0
currTime = 0
cap = cv.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img)
    if len(lm_list)!=0: 
        print(lm_list[4])

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)