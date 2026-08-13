import numpy as np
from time import time
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    drawing_utils, drawing_styles,
    HandLandmarker, HandLandmarksConnections, HandLandmarkerOptions,
)

model_path = 'models/hand_landmarker.task'
hand_opt = HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
HAND_DETECTOR = HandLandmarker.create_from_options(hand_opt)

def draw_hand_landmarks(frame, hand_landmarks):
    drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        HandLandmarksConnections.HAND_CONNECTIONS,
        drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
    )

class HandController:
    def __init__(self, max_durations=1.0):
        self.max_durations = max_durations
        self.start_time = None
        self.was_pinching = False

    def is_pinch(self, landmarks, threshold):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5
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

    @staticmethod
    def landmark_to_pixel(landmark, frame_shape):
        h, w = frame_shape[:2]

        return (
            int(landmark.x * w),
            int(landmark.y * h)
        )

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

        left_thumb = self.landmark_to_pixel(left_hand[4], frame_shape)
        left_index = self.landmark_to_pixel(left_hand[8], frame_shape)
        right_thumb = self.landmark_to_pixel(right_hand[4], frame_shape)
        right_index = self.landmark_to_pixel(right_hand[8], frame_shape)

        points = np.array([
            left_thumb, left_index, right_index, right_thumb
        ], dtype=np.int32)
        return points