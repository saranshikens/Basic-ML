import cv2 as cv
import mediapipe as mp


class FaceDetection:
    def __init__(self, detec_conf=0.5):
        self.detec_conf = detec_conf

        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFace.FaceDetection(self.detec_conf)

    def find_faces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        b_boxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                b_box_normalised = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                b_box = int(b_box_normalised.xmin*w), int(b_box_normalised.ymin*h), \
                        int(b_box_normalised.width*w), int(b_box_normalised.height*h)
                b_boxes.append([b_box, detection.score])
                
                if draw:
                    img = self.fancy_draw(img, b_box)

                    cv.putText(img, f'{int(detection.score[0]*100)}%', (b_box[0], b_box[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        return img, b_boxes
    
    def fancy_draw(self, img, b_box, length=30, thickness=5, rectangle_thickness=1):
        x, y, w, h = b_box
        x1, y1 = x+w, y+h

        cv.rectangle(img, b_box, (255,0,255), rectangle_thickness)

        # Top Left x,y
        cv.line(img, (x,y), (x+length,y), (255,0,255), thickness)
        cv.line(img, (x,y), (x,y+length), (255,0,255), thickness)
        # Top Right x1,y
        cv.line(img, (x1,y), (x1-length,y), (255,0,255), thickness)
        cv.line(img, (x1,y), (x1,y+length), (255,0,255), thickness)
        # Bottom Left x,y1
        cv.line(img, (x,y1), (x+length,y1), (255,0,255), thickness)
        cv.line(img, (x,y1), (x,y1-length), (255,0,255), thickness)
        # Bottom Right x1,y1
        cv.line(img, (x1,y1), (x1-length,y1), (255,0,255), thickness)
        cv.line(img, (x1,y1), (x1,y1-length), (255,0,255), thickness)

        return img

def main():
    cap = cv.VideoCapture(r'Advanced Computer Vision\Face Detection\Video\2.mp4')
    detector = FaceDetection()
    while True:
        success, img = cap.read()
        img, b_boxes = detector.find_faces(img)

        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__=="__main__":
    main()