import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(r'Advanced Computer Vision\Face Mesh\Video\2.mp4')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks: # faceLms is the face
            # Additionally, FACEMESH_TESSELATION, FACEMESH_IRISES
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                h,w,c = img.shape
                x,y = int(lm.x*w), int(lm.y*h)
                print(id,x,y)

    cv.imshow('Image', img)
    cv.waitKey(1)