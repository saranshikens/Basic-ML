import cv2 as cv
import mediapipe as mp
import Hand_Tracking_Module as htm
import os


wCam, hCam = 600, 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = r"Finger Counter\Images"
imgList = os.listdir(folderPath)
overLayList = []
for imgPath in imgList:
    image = cv.imread(f'{folderPath}/{imgPath}')
    overLayList.append(image)


detector = htm.handDetector(det_conf=0.75)

tipIDs = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)
    
    if len(lmList)!=0:
        fingers = []
        # Thumb of Left Hand
        if lmList[tipIDs[0]][1]<lmList[tipIDs[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        # Fingers of Left Hand
        for id in range(1,5):
            if lmList[tipIDs[id]][2]<lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)

        h,w,c = overLayList[totalFingers-1].shape
        img[0:h, 0:w] = overLayList[totalFingers-1]


    cv.imshow('Image', img)
    cv.waitKey(1)

