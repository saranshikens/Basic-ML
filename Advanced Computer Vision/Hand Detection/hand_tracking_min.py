import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # lm.x and lm.y are ratios by default, so we multiply them with w, h to retrieve coordinates
                centerX, centerY = int(lm.x*w), int(lm.y*h)
                print(id, centerX, centerY)
                # to track a certain part of hand, here the part which has id==0
                # if id==0:
                cv.circle(img, (centerX, centerY), 25, (255,0,255), cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    # display fps on the live feed
    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

    cv.imshow('Image', img)
    cv.waitKey(1)
    






