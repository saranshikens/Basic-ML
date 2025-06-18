import cv2 as cv
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self, mode=False, num_faces=2, detec_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.num_faces = num_faces
        self.detec_conf = detec_conf
        self.track_conf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.num_faces,
            min_detection_confidence=self.detec_conf,
            min_tracking_confidence=self.track_conf
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def find_facemesh(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
    
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h,w,c = img.shape
                    x,y = int(lm.x*w), int(lm.y*h)
                    cv.putText(img, str(id), (x,y), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
                    face.append([x,y])
                faces.append(face)

        return img, faces


def main():
    cap = cv.VideoCapture(r'Advanced Computer Vision\Face Mesh\Video\2.mp4')
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.find_facemesh(img)
        if len(faces)!=0:
            print(faces[0])
        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__=="__main__":
    main()