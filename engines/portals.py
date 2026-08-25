import time
import numpy as np
import cv2

from tools import HandController

class Inivisible:
    def __init__(
        self,
        empty_duration=2.0,
        min_person_ratio=0.01,
    ):
        self.background = None

        self.empty_duration = empty_duration
        self.min_person_ratio = min_person_ratio
        self.empty_since = None
        self.background_captured = False

        self.hand_controller = HandController()

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

    def process(self, frame: np.ndarray, **kwargs):
        hand_results = kwargs["hand_results"]
        seg_results = kwargs["seg_results"]

        human = self.is_human_exist(seg_results)
        self.update_background(frame, human)

        triggered = self.hand_controller.gesture_category(hand_results, "victory")
        if not triggered:
            return frame

        mask = seg_results.category_mask.numpy_view().squeeze()

        conditions = mask > 0
        foreground = frame.copy()

        blended = cv2.addWeighted(
            foreground,
            0.5, self.background, 0.5, 0
        )
        frame[conditions] = blended[conditions]
        return frame