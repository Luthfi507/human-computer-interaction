import numpy as np
from time import time
from utils import landmark_to_pixel

class HandController:
    def __init__(self, max_durations=1.0):
        self.max_durations = max_durations
        self.start_time = None
        self.was_pinching = False

    @staticmethod
    def distance(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    def is_pinch(self, hand_results, threshold):
        distances = []
        for landmarks in hand_results:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            distances.append(self.distance(thumb_tip, index_tip))

        distance = min(distances)
        return distance <= threshold

    def update(self, landmarks, threshold=0.05):
        current_time = time()
        triggered = False

        is_pinching = self.is_pinch(landmarks, threshold)

        # pinch is started
        if is_pinching and not self.was_pinching:
            self.start_time = current_time

        elif not is_pinching and self.was_pinching:
            if self.start_time is not None:
                duration = current_time - self.start_time

                if duration < self.max_durations:
                    triggered = True

            self.start_time = None

        self.was_pinching = is_pinching
        return triggered
    
    def get_poly_point(self, results, frame_shape):
        if len(results.hand_landmarks) < 2:
            return

        left_hand = None
        right_hand = None

        for landmarks, handedness in zip(
            results.hand_landmarks, results.handedness
        ):
            hand_name = handedness[0].category_name

            if hand_name == "Left":
                left_hand = landmarks
            elif hand_name == "Right":
                right_hand = landmarks

        if left_hand is None or right_hand is None:
            return

        left_thumb = landmark_to_pixel(left_hand[4], frame_shape)
        left_index = landmark_to_pixel(left_hand[8], frame_shape)
        right_thumb = landmark_to_pixel(right_hand[4], frame_shape)
        right_index = landmark_to_pixel(right_hand[8], frame_shape)

        points = np.array([
            left_thumb, left_index, right_index, right_thumb
        ], dtype=np.int32)
        return points

    