import cv2
import numpy as np

from tools import FILTERS, HandController
from utils import draw_hand_landmarks
hand_controller = HandController()

class Pinch:
    def __init__(self, threshold = 0.05):
        self.threshold = threshold
        self.current_filter = 0
        
    def update_filter(self, hand_landmarks):
        pinching = hand_controller.update(hand_landmarks)

        if pinching:
            self.current_filter = (
                self.current_filter + 1
            ) % len(FILTERS)

class Pipeline(Pinch):
    def __init__(
            self,
            draw_landmarks=False,
            threshold=0.05,
            smoothing=0.5
        ):
        super().__init__(threshold)
        self.draw_landmarks = draw_landmarks
        self.smoothing = smoothing
        self.prev_points = None

    @staticmethod
    def landmark_to_pixel(landmark, frame_shape):
        h, w = frame_shape[:2]
        return (int(landmark.x * w), int(landmark.y * h))

    def smooth_points(self, points):
        if self.prev_points is None:
            self.prev_points = points.copy()
            return points

        points = (
            self.smoothing * points
            + (1.0 - self.smoothing) * self.prev_points
        )
        self.prev_points = points.copy()
        return points

    @staticmethod
    def order_polygon(points):
        center = np.mean(points, axis=0)

        angles = np.arctan2(
            points[:, 1] - center[1],
            points[:, 0] - center[0]
        )

        order = np.argsort(angles)
        return points[order]

    def get_poly_point(self, results, frame):
        frame_shape = frame.shape
        if len(results.hand_landmarks) < 2:
            self.prev_points = None
            return

        left_hand = None
        right_hand = None

        for landmarks, handedness in zip(
            results.hand_landmarks, results.handedness
        ):
            if self.draw_landmarks:
                draw_hand_landmarks(frame, landmarks)
            hand_name = handedness[0].category_name

            if hand_name == "Left":
                left_hand = landmarks
            elif hand_name == "Right":
                right_hand = landmarks

        if left_hand is None or right_hand is None:
            self.prev_points = None
            return

        left_thumb = self.landmark_to_pixel(left_hand[4], frame_shape)
        left_index = self.landmark_to_pixel(left_hand[8], frame_shape)
        right_thumb = self.landmark_to_pixel(right_hand[4], frame_shape)
        right_index = self.landmark_to_pixel(right_hand[8], frame_shape)

        points = np.array([
            left_thumb, left_index, right_index, right_thumb
        ], dtype=np.int32)

        points = self.smooth_points(points)
        return np.round(points).astype(np.int32)

    def polly_process(self, frame: np.ndarray, hand_results):
        points = self.get_poly_point(hand_results, frame)
        if points is None:
            return frame

        self.update_filter(hand_results.hand_landmarks)        
        x, y, w, h = cv2.boundingRect(points)

        roi = frame[y:y+h, x:x+w]
        filtered_roi = FILTERS[self.current_filter](roi)

        local_points = points - np.array([x, y])
        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(mask, [local_points], 255)
        roi[mask > 0] = filtered_roi[mask > 0]
        frame[y:y+h, x:x+w] = roi

        return frame