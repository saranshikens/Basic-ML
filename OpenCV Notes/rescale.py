import cv2 as cv

# Functions to Rescale
def rescaleFrame(frame, scale=0.2):
    # Images, Videos, and Live Videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # Live Videos
    capture.set(3, width)
    capture.set(4, height)


# Rescaling Image
img = cv.imread(r'Photos\cat_large.jpg')
cv.imshow('Cat', img)

img_resized = rescaleFrame(img)
cv.imshow('Resized Cat', img_resized)


# Rescaling Video
capture = cv.VideoCapture(r'Videos\dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Resized Video', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'): # if 'd' is pressed, stop the video
        break

capture.release()
cv.destroyAllWindows() 