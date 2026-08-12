from time import time
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceDetectorOptions, FaceDetector
)

from tools.filters import FILTERS
from utils.hand_tracking import PinchController, HAND_DETECTOR, is_pinch, draw_hand_landmarks

opt = FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path="models/blaze_face_full_range.tflite"))
detector = FaceDetector.create_from_options(opt)

def process_face(frame, face_results, current_filter):
    for result in face_results.detections:
        bboxes = result.bounding_box

        x1, y1 = bboxes.origin_x, bboxes.origin_y
        x2, y2 = bboxes.origin_x + bboxes.width, bboxes.origin_y + bboxes.height

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = FILTERS[current_filter](roi)

    return frame

def process_hands(frame, hand_results):
    pinching = False

    for landmarks in hand_results.hand_landmarks:
        draw_hand_landmarks(frame, landmarks)

        if is_pinch(landmarks):
            pinching = True

    return pinching

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera detected")

    current_filter = 0
    pinch_controller = PinchController()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
        face_results = detector.detect(mp_image)
        hand_results = HAND_DETECTOR.detect(mp_image)

        pinching = process_hands(frame, hand_results)
        if pinch_controller.update(pinching):
            current_filter = (
                current_filter + 1
            ) % len(FILTERS)

        frame = process_face(frame, face_results, current_filter)

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()