import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(r'Advanced Computer Vision\Face Detection\Video\2.mp4')

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection()

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            b_box_normalised = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            b_box = int(b_box_normalised.xmin*w), int(b_box_normalised.ymin*h), \
                    int(b_box_normalised.width*w), int(b_box_normalised.height*h)
            cv.rectangle(img, b_box, (255,0,255), 2)
            cv.putText(img, f'{int(detection.score[0]*100)}%', (b_box[0], b_box[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)        

    cv.imshow('Image', img)
    cv.waitKey(1)