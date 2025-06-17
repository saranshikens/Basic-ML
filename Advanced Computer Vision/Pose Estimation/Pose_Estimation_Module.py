import cv2 as cv
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, mode=False, upper_body=False, smooth=True, detec_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detec_conf = detec_conf
        self.track_conf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            enable_segmentation=False,
            min_detection_confidence=self.detec_conf,
            min_tracking_confidence=self.track_conf
        )

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
    
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    def findPosition(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                centerX, centerY = int(lm.x*w), int(lm.y*h)
                lm_list.append([id, centerX, centerY])

        return lm_list


def main():
    cap = cv.VideoCapture(r'Advanced Computer Vision\Pose Estimation\Videos\1.mp4')

    detector = PoseDetector()

    while True:
        success, img = cap.read()
        detector.findPose(img)
        lm_list = detector.findPosition(img)
        print(lm_list)

        cv.imshow('Image', img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()