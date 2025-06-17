import cv2 as cv
import numpy as np

# Training

features = []
labels = []

face_recognizer = cv.face.LBPHFaceRecognizer_create()
# saving the model, this will create a separate file for the model
face_recognizer.save('face_trained.yml')

face_recognizer.train(features, labels)
np.save('features.npy', features)
np.save('labels.npy', labels)



# Predicting

haar_cascade = cv.CascadeClassifier(r'OpenCV Notes\haar_face.xml')

features = np.load('features.npy')
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'OpenCV Notes\Photos\lady.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    