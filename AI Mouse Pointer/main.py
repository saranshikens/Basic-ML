import cv2 as cv
import mediapipe as mp
import numpy as np
import Hand_Tracking_Module as htm
import autopy


################################
wCam, hCam = 640, 480
frameR = 50 # frame reduction
smoothening = 5
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0
################################

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(3, hCam)

detector = htm.handDetector(maxHands=1)
wScreen, hScreen = autopy.screen.size()

while True:
    # Find hand landmarks
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList, bBox = detector.find_position(img)

    # Get the tip of index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which finger is up
        fingers = detector.fingersUp()
        cv.rectangle(img, (frameR, frameR), (wCam-frameR,hCam-frameR), (255,0,255),2)

        # Only index finger -> Moving mode
        if fingers[1]==1 and fingers[2]==0:
    
            # Convert coordinates
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScreen))
            y3 = np.interp(x1, (frameR,hCam-frameR), (0,hScreen))

            # Smoothen values
            currLocX = prevLocX + (x3 - prevLocX)/smoothening
            currLocY = prevLocY + (x3 - prevLocY)/smoothening

            # Move mouse
            autopy.mouse.move(wScreen-currLocX,currLocY) # wSreen-x3 inverts the mouse pointer
            cv.circle(img, (x1,y1), 15, (255,0,255), cv.FILLED)
            prevLocX, prevLocY = currLocX, currLocY

        # Check if both are up : Clicking mode
        if fingers[1]==1 and fingers[2]==1:
            # Find distance between the tips
            length, img, lineInfo = detector.findDistance(8,12,img)
            # Click mouse if distance is short
            if length<40:
                cv.circle(img, (lineInfo[4],lineInfo[5]), 15, (0,255,0), cv.FILLED)
                autopy.mouse.click()


    cv.imshow('Image', img)
    cv.waitKey(1)