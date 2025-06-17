import cv2 as cv
import mediapipe as mp
import time
import Pose_Estimation_Module as pem


cap = cv.VideoCapture(r'Advanced Computer Vision\Pose Estimation\Videos\1.mp4')

detector = pem.PoseDetector()

while True:
    success, img = cap.read()
    detector.findPose(img)
    lm_list = detector.findPosition(img)
    if len(lm_list)!=0:
        print(lm_list)

    cv.imshow('Image', img)
    cv.waitKey(1)