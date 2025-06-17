import cv2 as cv

# Reading Images

img = cv.imread(r'Photos\cat_large.jpg')
cv.imshow('Cat', img) 

cv.waitKey(0)

# Reading Videos

capture = cv.VideoCapture(r'Videos\dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'): # if 'd' is pressed, stop the video
        break

capture.release()
cv.destroyAllWindows() 