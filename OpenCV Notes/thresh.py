import cv2 as cv

img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# if pixel < 150 -> pixel = 0
# if pixel > 150 -> pixel = 255
cv.imshow('Simple Thresholded', thresh)

# Inversing Simple Threshold
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inversed', thresh_inv)

# Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
# automacically chooses threshold value
# optimises mean of a neighbour of some size
# here, size = 11
# c is subtracted from the mean, here c = 3
cv.imshow('Adaptive Thresholded', adaptive_thresh)


cv.waitKey(0)