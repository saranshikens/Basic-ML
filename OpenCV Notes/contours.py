import cv2 as cv
import numpy as np

img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img) 

blank = np.zeros(img.shape, dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# without blur -> 479 contours
# with blur -> 62 contours

canny = cv.Canny(blur, 125, 175)


ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) 
cv.imshow('Thresh', thresh)
# binarizes an image, if pixel < 125, pixel = 0 else pixel = 255
# 255 contours


contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# to use thresh, type thresh in place of canny
# RETR -> all the hierarchical contours
# EXTERNAL -> external contours
# LIST -> all contours
# CHAIN_APPROX_NONE does nothing
# CHAIN_APPROX_SIMPLE compresses the contours into those that make sense
# if there is a line, NONE returns all the points on that line
# SIMPLE returns the two end points on that line
print(len(contours))

cv.drawContours(blank, contours, -1, (0,0,255), 1)
# -1 -> all contours
# 2 -> thickness
cv.imshow('Contour', blank)


cv.waitKey(0)