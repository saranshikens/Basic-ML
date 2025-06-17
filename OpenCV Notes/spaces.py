import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img)

plt.imshow(img)
# matplotlib shows an image as RGB, so colors will be inverted
plt.show()

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray) 

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv) 

# BGR to LAB or L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('Lab', lab) 

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# cannot convert grayscale to HSV, convert to BGR first, then convert to HSV
cv.waitKey(0)