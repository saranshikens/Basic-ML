import cv2 as cv

img = cv.imread(r'Photos\cat.jpeg')
cv.imshow('Cat', img) 

# Converting to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Cat', gray) 

# Blurring
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT) # (3,3) is kernel, has to odd
cv.imshow('Blur', blur)

# Edge Detector
canny = cv.Canny(blur, 125, 175)
cv.imshow('Edges', canny) 

# Dilating 
dilated = cv.dilate(canny, (7,7), iterations=3) # (7,7) is kernel
cv.imshow('Dilated', dilated) 

# Eroding a Dilation
eroded = cv.erode(dilated, (7,7), iterations=3) # (3,3) is kernel
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_LINEAR) 
# interpolation=cv.INTER_AREA, if making an image smaller and cv.INTER_LINEAR, SQUARE, CUBIC to make it larger.
# Quality increases by using SQUARE over LINEAR
cv.imshow('Resized', resized) 

# Cropping
cropped = img[10:20, 30:40]
cv.imshow('Cropped', cropped) 

cv.waitKey(0)