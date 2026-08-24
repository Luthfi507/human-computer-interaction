import time
import cv2
import numpy as np
import mediapipe as mp

from utils import SEGMENTER, HAND_DETECTOR
from tools import HandController

class Inivisible:
    def __init__(
        self,
        camera=0,
        empty_duration=2.0,
        min_person_ratio=0.01,
    ):
        self.camera = camera
        self.cap = None
        self.background = None

        self.empty_duration = empty_duration
        self.min_person_ratio = min_person_ratio
        self.empty_since = None
        self.background_captured = False

        self.hand_controller = HandController()

    def get_camera(self):
        self.cap = cv2.VideoCapture(self.camera)

        if not self.cap.isOpened():
            raise RuntimeError("No camera detected")

    def is_human_exist(self, result):
        mask = result.category_mask.numpy_view().squeeze()
        person_mask = mask > 0
        person_ratio = np.mean(person_mask)

        human = person_ratio >= self.min_person_ratio
        return human

    def update_background(self, frame, human):
        now = time.monotonic()

        if human:
            self.empty_since = None
            self.background_captured = False
            return False

        if self.empty_since is None:
            self.empty_since = now
            return False

        elapsed = now - self.empty_since

        if (
            elapsed >= self.empty_duration
            and not self.background_captured
        ):
            self.background = frame.copy()
            self.background_captured = True

            print("Background updated.")
            return True

        return False

    def detect_pinch(self, hand_result):
        if not hand_result.hand_landmarks:
            return False

        landmarks = hand_result.hand_landmarks
        pinching = self.hand_controller.is_pinch(landmarks, 0.05)
        return pinching

    def run(self):
        self.get_camera()

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

            seg_result = SEGMENTER.segment(mp_image)
            hand_result = HAND_DETECTOR.detect(mp_image)

            human = self.is_human_exist(seg_result)

            self.update_background(frame, human)
            output = frame.copy()

            pinch_triggered = self.detect_pinch(hand_result)
            if self.background is not None and pinch_triggered:
                output = self.background

            cv2.imshow("camera", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipe = Inivisible(
        camera=0,
        empty_duration=2.0,
        min_person_ratio=0.01,
    )

    pipe.run()