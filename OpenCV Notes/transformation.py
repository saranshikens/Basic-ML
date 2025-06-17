import cv2 as cv
import numpy as np

img = cv.imread(r'Photos\cat.jpeg')
cv.imshow('Cat', img) 


# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)
# negative x -> left
# negative y -> up
# positive x -> right
# positive y -> down

translated = translate(img, 50, 50)
cv.imshow('Translated', translated)


# Rotation
def rotate(img, angle, pivotPoint):
    (height, width) = img.shape[:2]

    if pivotPoint is None:
        pivotPoint = (width//2, height//2)

    rotMap = cv.getRotationMatrix2D(pivotPoint, angle, 1.0) # 1.0 is scaling factor
    dimensions = (width, height)

    return cv.warpAffine(img, rotMap, dimensions)

rotated = rotate(img, 45, None) # -45 for clockwise
cv.imshow('Rotated', rotated)


# Flipping
flip = cv.flip(img, 0)
# 0-> vertical flip
# 1 -> horizontal flip
# -1 -> both
cv.imshow('Flip', flip)

cv.waitKey(0)