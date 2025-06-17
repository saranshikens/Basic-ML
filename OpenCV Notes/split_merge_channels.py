import cv2 as cv
import numpy as np

img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')


# b,g,r will be grayscale
b,g,r = cv.split(img)
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged', merged)


# to visualize the color distribution
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

cv.waitKey(0)