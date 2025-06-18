import cv2 as cv
import mediapipe as mp
import Hand_Tracking_Module as htm
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480


cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(det_conf=0.7)


# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# Cast the interface to IAudioEndpointVolume
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]


while True:
    success, img = cap.read()
    img = detector.find_hands(img)

    lmList = detector.find_position(img, draw=False)
    if len(lmList)!=0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        centerX, centerY = (x1+x2)//2, (y1+y2)//2

        cv.circle(img, (x1,y1), 10, (255,0,255), cv.FILLED)
        cv.circle(img, (x2,y2), 10, (255,0,255), cv.FILLED)
        cv.circle(img, (centerX,centerY), 10, (255,0,255), cv.FILLED)
        cv.line(img, (x1,y1), (x2,y2), (255,0,255), 3)

        length = math.hypot(x2-x1,y2-y1)

        # length Range 50-300
        # volume Range -65 - 0

        vol = np.interp(length, [50,300], [minVol, maxVol])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv.circle(img, (centerX,centerY), 15, (0,255,0), cv.FILLED)

    cv.imshow('Image', img)
    cv.waitKey(1)
