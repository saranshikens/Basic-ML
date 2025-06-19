import cv2 as cv
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, det_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.det_conf = det_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.track_conf
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIDs = [4,8,12,16,20]

    def find_hands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def find_position(self, img, hand_num=0, draw=True):

        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                centerX, centerY = int(lm.x*w), int(lm.y*h)
                self.lm_list.append([id, centerX, centerY])
                if draw:
                    cv.circle(img, (centerX, centerY), 3, (255,0,255), cv.FILLED)

        return self.lm_list
    
    def fingersUp(self):
        fingers = []
        # Thumb of Left Hand
        if self.lm_list[self.tipIDs[0]][1]<self.lm_list[self.tipIDs[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        # Fingers of Left Hand
        for id in range(1,5):
            if self.lm_list[self.tipIDs[id]][2]<self.lm_list[self.tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers



def main():
    prevTime = 0
    currTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list)!=0: 
            print(lm_list[4])

        currTime = time.time()
        fps = 1/(currTime-prevTime)
        prevTime = currTime

        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

        cv.imshow('Image', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()