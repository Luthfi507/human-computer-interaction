import cv2
import numpy as np

from tools import FILTERS, HandController
hand_controller = HandController()

class Pinch:
    def __init__(self, threshold = 0.05):
        self.threshold = threshold
        self.current_filter = 0
        
    def update_filter(self, hand_landmarks):
        pinching = False

        for landmarks in hand_landmarks:
            pinching = hand_controller.update(landmarks, self.threshold)

        if pinching:
            self.current_filter = (
                self.current_filter + 1
            ) % len(FILTERS)

class Pipeline(Pinch):
    def __init__(self, threshold=0.05):
        super().__init__(threshold)

    @staticmethod
    def landmark_to_pixel(landmark, frame_shape):
        h, w = frame_shape[:2]
        return (int(landmark.x * w), int(landmark.y * h))

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

    def polly_process(self, frame: np.ndarray, hand_results):
        points = self.get_poly_point(hand_results, frame.shape)
        if points is None:
            return frame

        self.update_filter(hand_results.hand_landmarks)        
        x, y, w, h = cv2.boundingRect(points)

        roi = frame[y:y+h, x:x+w]
        filtered_roi = FILTERS[self.current_filter](roi)

        local_points = points - np.array([x, y])
        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(mask, [local_points], 255)
        roi[mask == 255] = filtered_roi[mask == 255]
        frame[y:y+h, x:x+w] = roi

        return frame