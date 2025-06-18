import cv2 as cv
import mediapipe as mp
import os
import Hand_Tracking_Module as htm

folderPath = r"AI Painter\Images"
imgList = os.listdir(folderPath)
overLayList = []

for imgPath in imgList:
    img = cv.imread(f'{folderPath}/{imgPath}')
    overLayList.append(img)

header = overLayList[0]

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(det_conf=0.85)

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
            cv.rectangle(img, (x1,y1-25), (x2,y2+25), (255,0,255), cv.FILLED)

        # Drawing Mode
        if fingers[1] and fingers[2]==False:
            cv.circle(img, (x1,y1), 15, (255,0,255), cv.FILLED)

    # Setting the header image
    img[0:120, 0:1280] = header

    cv.imshow('Image', img)
    cv.waitKey(1)


    