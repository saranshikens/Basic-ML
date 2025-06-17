import cv2 as cv

img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img)


# Averaging
average = cv.blur(img, (3,3))
cv.imshow('Average', average)

# Gaussian
gaussian = cv.GaussianBlur(img, (3,3), 0) 
# third parameter is sigmaX i.e. standard deviation in X direction
cv.imshow('Gaussian', gaussian)

# Median 
median = cv.medianBlur(img, 3)
# 3 is diameter of pixel neighbourhood
# for salt-and-pepper noise
cv.imshow('Median', median)

# Bilateral
bilateral = cv.bilateralFilter(img, 3, 15, 15)
# retains edges
# third param -> sigma color : larger sigma = no color in neighbourhood to be considered for blurring
# fourth param -> sigma space : larger sigma = pixels farther from central pixel will be considered
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)