import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread(r'Photos\cats_1.jpeg')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blank = np.zeros(img.shape[:2], dtype='uint8')
circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 30, 255, -1)
mask = cv.bitwise_and(gray, gray, mask = circle)

gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256])
# second param -> color channels (for indexing)
# third param -> mask
# fourth para -> # of bins
# fifth param -> range of values
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

# Color Histogram
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
colors = ('b','g','r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()


cv.waitKey(0)