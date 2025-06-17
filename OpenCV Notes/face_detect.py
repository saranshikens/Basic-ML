import cv2 as cv

img = cv.imread(r'OpenCV Notes\Photos\lady.jpeg')
# cv.imshow('Lady', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)


haar_cascade = cv.CascadeClassifier(r'OpenCV Notes\haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
# minNeighbors specifies how many neighbors each candidate rectangle should have to retain it

print(len(faces_rect))

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)