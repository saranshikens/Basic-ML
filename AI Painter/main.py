import cv2 as cv
import mediapipe as mp
import os
import Hand_Tracking_Module as htm
import numpy as np


brushThickness = 15
eraserThickness = 50


folderPath = r"AI Painter\Images"
imgList = os.listdir(folderPath)
overLayList = []

for imgPath in imgList:
    img = cv.imread(f'{folderPath}/{imgPath}')
    overLayList.append(img)

header = overLayList[0]
drawColor = (255,0,255)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(det_conf=0.85)
xprev, yprev = 0, 0

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    # Import image
    success, img = cap.read()
    img = cv.flip(img, 1)
    
    # Find Landmarks
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    if len(lmList)!=0:
        # tip of index and middle finger
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]

        # Check which finger is up
        fingers = detector.fingersUp()

        # Selection Mode
        if fingers[1] and fingers[2]:
            xprev, yprev = 0, 0
            if y1<125:
                if 250<x1<450:
                    header = overLayList[0]
                    drawColor = (255,0,255)
                elif 550<x1<750:
                    header = overLayList[1]
                    drawColor = (255,0,0)
                elif 800<x1<950:
                    header = overLayList[2]
                    drawColor = (0,255,0)
                elif 1050<x1<1200:
                    header = overLayList[3]
                    drawColor = (0,0,0)
            cv.rectangle(img, (x1,y1-25), (x2,y2+25), drawColor, cv.FILLED)

        # Drawing Mode
        if fingers[1] and fingers[2]==False:
            cv.circle(img, (x1,y1), 15, (255,0,255), cv.FILLED)
            if xprev==0 and yprev==0:
                xprev, yprev = x1, y1

            if drawColor==(0,0,0):
                cv.line(img, (xprev, yprev), (x1,y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xprev, yprev), (x1,y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xprev, yprev), (x1,y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xprev, yprev), (x1,y1), drawColor, brushThickness)
            xprev, yprev = x1, y1


    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:120, 0:1280] = header

    img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv.imshow('Image', img)
    #cv.imshow('Canvas', imgCanvas)
    cv.waitKey(1)


    