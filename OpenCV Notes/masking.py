import cv2 as cv
import numpy as np


img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 50, 255, -1)
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked', masked)


cv.waitKey(0)