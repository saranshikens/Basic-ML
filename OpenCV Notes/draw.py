import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# Coloring an Image
blank[:] = 0,255,0
cv.imshow('Green', blank)

# Painting a Portion of Image
blank[200:300, 300:400] = 255,0,0
cv.imshow('Portion', blank)

# Drawing a Rectangle
cv.rectangle(blank, (0,0), (250,250), (0,0,255), thickness=2) # to fill the rectangle, thickness=-1 OR thickness=cv.FILLED
cv.imshow('Rectangle', blank)

# Drawing a Circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=3)
cv.imshow('Circle', blank)

# Drawing a Line
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=3)
cv.imshow('Line', blank)

# Putting a Text
cv.putText(blank, 'BALLS', (225,225), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
cv.imshow('Text', blank)

cv.waitKey(0)