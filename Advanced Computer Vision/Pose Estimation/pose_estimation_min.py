import cv2 as cv
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture(r'Advanced Computer Vision\Pose Estimation\Videos\1.mp4')

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(id, lm)
            centerX, centerY = int(lm.x*w), int(lm.y*h)

    cv.imshow('Image', img)
    cv.waitKey(1)