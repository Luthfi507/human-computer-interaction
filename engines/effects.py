import cv2
import numpy as np
import random

from tools import FILTERS, HandController
from utils import landmark_to_pixel, draw_hand_landmarks, hand_landmark_separator
hand_controller = HandController()

class Pinch:
    def __init__(
            self, 
            draw_landmarks=False,
            threshold = 0.05
        ):
        self.draw_landmarks = draw_landmarks
        self.threshold = threshold
        self.current_filter = random.choice(FILTERS)
        
    def update_filter(self, hand_landmarks):
        if len(hand_landmarks) < 1:
            return
        
        pinching = hand_controller.update(hand_landmarks, self.threshold)
        if pinching:
            self.current_filter = random.choice(FILTERS)

class RectanglePipeline(Pinch):
    def __init__(self, draw_landmarks=False, threshold=0.05):
        super().__init__(draw_landmarks, threshold)
        self.committed_regions = []

    def get_points(self, frame, hand_results):
        left_hand, right_hand = hand_landmark_separator(frame, self.draw_landmarks, hand_results)
        if left_hand is None or right_hand is None:
            return

        left_index = landmark_to_pixel(left_hand[8], frame)
        right_index = landmark_to_pixel(right_hand[8], frame)

        x1 = min(left_index[0], right_index[0])
        y1 = min(left_index[1], right_index[1])
        x2 = max(left_index[0], right_index[0])
        y2 = max(left_index[1], right_index[1])
        return x1, y1, x2, y2

    def filter_region(self, frame, points, current_filter):
        x1, y1, x2, y2 = points
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return

        filtered_roi = current_filter(roi)
        frame[y1:y2, x1:x2] = filtered_roi

    def process(self, frame: np.ndarray, **kwargs):
        hand_results = kwargs['hand_results']
        for region in self.committed_regions:
            self.filter_region(
                frame,
                region["points"],
                region["filter"]
            )

        if len(hand_results.hand_landmarks) < 2:
            return frame
        
        points = self.get_points(frame, hand_results)

        if not points:
            return frame

        previous_filter = self.current_filter
        self.update_filter(hand_results.hand_landmarks)

        if self.current_filter != previous_filter:
            self.committed_regions.append({
                "points": points,
                "filter": previous_filter
            })

        self.filter_region(
            frame,
            points,
            self.current_filter
        )

        return frame

class PolyPipeline(Pinch):
    def __init__(
            self,
            draw_landmarks=False,
            threshold=0.05,
            smoothing=0.5
        ):
        super().__init__(draw_landmarks, threshold)
        self.smoothing = smoothing
        self.prev_points = None

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

    def get_poly_point(self, hand_results, frame):
        if len(hand_results.hand_landmarks) < 2:
            self.prev_points = None
            return
        
        left_hand, right_hand = hand_landmark_separator(frame, self.draw_landmarks, hand_results)
        if left_hand is None or right_hand is None:
            self.prev_points = None
            return

        left_thumb = landmark_to_pixel(left_hand[4], frame)
        left_index = landmark_to_pixel(left_hand[8], frame)
        right_thumb = landmark_to_pixel(right_hand[4], frame)
        right_index = landmark_to_pixel(right_hand[8], frame)

        points = np.array([
            left_thumb, left_index, right_index, right_thumb
        ], dtype=np.int32)

        points = self.smooth_points(points)
        return np.round(points).astype(np.int32)

    def process(self, frame: np.ndarray, **kwargs):
        hand_results = kwargs['hand_results']
        points = self.get_poly_point(hand_results, frame)
        if points is None:
            return frame

        self.update_filter(hand_results.hand_landmarks)        
        x, y, w, h = cv2.boundingRect(points)

        roi = frame[y:y+h, x:x+w]
        filtered_roi = self.current_filter(roi)

        local_points = points - np.array([x, y])
        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(mask, [local_points], 255)
        roi[mask > 0] = filtered_roi[mask > 0]
        frame[y:y+h, x:x+w] = roi

        return frame

class CirclePipeline(Pinch):
    def __init__(self, draw_landmarks=False, threshold=0.05, smoothing=0.5):
        super().__init__(draw_landmarks, threshold)
        self.smoothing = smoothing
        self.prev_radius = None
        self.prev_center = None

    def smooth_circle(self, center, radius):
        center = np.array(
            center, np.float32
        )

        if self.prev_center is None:
            self.prev_center = center.copy()
            self.prev_radius = radius

        center = self.smoothing * center + (1 - self.smoothing) * self.prev_center
        radius = self.smoothing * radius + (1 - self.smoothing) * self.prev_radius

        self.prev_center = center.copy()
        self.prev_radius = radius

        return center, radius

    def get_circle(self, hand_results, frame):
        if len(hand_results.hand_landmarks) < 2:
            self.prev_radius = None
            self.prev_center = None
            return
        
        left_hand, right_hand = hand_landmark_separator(frame, self.draw_landmarks, hand_results)
        if left_hand is None or right_hand is None:
            self.prev_center = None
            self.prev_radius = None
            return

        left_index = landmark_to_pixel(left_hand[8], frame)
        right_index = landmark_to_pixel(right_hand[8], frame)

        left_index = np.array(left_index, dtype=np.float32)
        right_index = np.array(right_index, dtype=np.float32)

        center = (left_index + right_index) / 2
        radius = np.linalg.norm(left_index - right_index) / 2

        center, radius = self.smooth_circle(center, radius)

        center = tuple(np.round(center).astype(np.int32))
        radius = int(round(radius))

        return center, radius

    def process(self, frame: np.ndarray, **kwargs):
        hand_results = kwargs['hand_results']
        circle = self.get_circle(hand_results, frame)

        if circle is None:
            return frame

        center, radius = circle

        if radius <= 1:
            return frame

        self.update_filter(hand_results.hand_landmarks)

        cx, cy = center
        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)

        x2 = min(frame_w, cx + radius)
        y2 = min(frame_h, cy + radius)

        roi = frame[y1:y2,x1:x2]

        filtered_roi = self.current_filter(roi.copy())

        local_center = (cx - x1,cy - y1)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        cv2.circle(mask, local_center, radius, 255, -1)
        roi[mask > 0] = filtered_roi[mask > 0]

        return frame

class SpotlightPipeline(Pinch):
    def __init__(self, draw_landmarks=False, threshold=0.05, smoothing=0.5):
        super().__init__(draw_landmarks, threshold)
        self.smoothing = smoothing
        self.prev_center = None
        self.prev_radius = None

    def smooth_spotlight(self, center, radius):
        center = np.array(center, dtype=np.float32)

        if self.prev_center is None:
            self.prev_center = center.copy()
            self.prev_radius = radius
            return center, radius

        center = self.smoothing * center+ (1.0 - self.smoothing) * self.prev_center
        radius = self.smoothing * radius+ (1.0 - self.smoothing) * self.prev_radius

        self.prev_center = center.copy()
        self.prev_radius = radius
        return center, radius

    def get_spotlight(self, landmarks, frame):
        if self.draw_landmarks:
            draw_hand_landmarks(frame, landmarks)

        index_tip = landmark_to_pixel(landmarks[8], frame)
        thumb_tip = landmark_to_pixel(landmarks[4], frame)

        index_tip = np.array(index_tip, dtype=np.float32)
        thumb_tip = np.array(thumb_tip, dtype=np.float32)

        center = index_tip
        radius = np.linalg.norm(thumb_tip - index_tip)

        center, radius = self.smooth_spotlight(center, radius)

        center = tuple(np.round(center).astype(np.int32))
        radius = int(round(radius))
        return center, radius

    def process(self, frame: np.ndarray, **kwargs):
        hand_results = kwargs['hand_results']
        if len(hand_results.hand_landmarks) == 0:
            self.prev_center = None
            self.prev_radius = None
            return frame

        spotlight = self.get_spotlight(hand_results.hand_landmarks[0], frame)
        if spotlight is None:
            return frame

        center, radius = spotlight

        if radius <= 1:
            return frame

        self.update_filter(hand_results.hand_landmarks)

        cx, cy = center

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)

        x2 = min(frame_w, cx + radius)
        y2 = min(frame_h, cy + radius)

        roi = frame[y1:y2, x1:x2]
        if roi.size  == 0:
            return frame

        filtered_roi = self.current_filter(roi.copy())

        local_center = (cx - x1, cy - y1)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        cv2.circle(mask, local_center, radius, 255, -1)
        roi[mask > 0] = filtered_roi[mask > 0]
        return frame

class SpotlightBackground(SpotlightPipeline):
    def __init__(self, draw_landmarks=False, threshold=0.05):
        super().__init__(draw_landmarks, threshold)
        self.background_filter = None

    def update_background(self, landmarks):
        is_pinching = hand_controller.update([landmarks], self.threshold)
        if is_pinching:
            self.background_filter = random.choice(FILTERS)

    def process(self, frame, **kwargs):
        hand_results = kwargs["hand_results"]
        if len(hand_results.hand_landmarks) == 0:
            self.prev_center = None
            self.prev_radius = None

            if self.background_filter is not None:
                frame  = self.background_filter(frame.copy())
            return frame

        left_hand, right_hand = hand_landmark_separator(frame, self.draw_landmarks, hand_results)
        if right_hand is not None:
            self.update_background(right_hand)

        if self.background_filter is not None:
            frame = self.background_filter(frame.copy())

        if left_hand is None:
            return frame
        
        spotlight = self.get_spotlight(left_hand, frame)
        if spotlight is None:
            return frame

        center, radius = spotlight

        if radius <= 1:
            return frame

        self.update_filter([left_hand])

        cx, cy = center

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)

        x2 = min(frame_w, cx + radius)
        y2 = min(frame_h, cy + radius)

        roi = frame[y1:y2, x1:x2]
        if roi.size  == 0:
            return frame

        filtered_roi = self.current_filter(roi.copy())

        local_center = (cx - x1, cy - y1)
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)

        cv2.circle(mask, local_center, radius, 255, -1)
        roi[mask > 0] = filtered_roi[mask > 0]
        return frame

class SelfieSegmentation(Pinch):
    def __init__(self, draw_landmarks=False, threshold=0.05):
        super().__init__(draw_landmarks, threshold)

    def process(self, frame, **kwargs):
        hand_results = kwargs['hand_results']
        seg_results = kwargs['seg_results']
        if self.draw_landmarks:
            for landmarks in hand_results.hand_landmarks:
                draw_hand_landmarks(frame, landmarks)

        self.update_filter(hand_results.hand_landmarks)        
        mask = seg_results.category_mask.numpy_view().squeeze()
        person_mask = mask > 0

        filtered_frame = self.current_filter(frame.copy())
        frame[person_mask] = filtered_frame[person_mask]

        return frame