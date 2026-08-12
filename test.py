from time import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python.vision import (
    drawing_utils, drawing_styles,
    FaceDetectorOptions, FaceDetector
)

from tools.filters import FILTERS

opt = FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path="models/blaze_face_full_range.tflite"))
detector = FaceDetector.create_from_options(opt)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera detected")

    last_filter_time = time()
    current_filter = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_time = time()
        if current_time - last_filter_time >= 1:
            current_filter = (current_filter + 1) % len(FILTERS)
            last_filter_time = current_time

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
        results = detector.detect(mp_image)

        for result in results.detections:
            bboxes = result.bounding_box

            x1, y1 = bboxes.origin_x, bboxes.origin_y
            x2, y2 = bboxes.origin_x + bboxes.width, bboxes.origin_y + bboxes.height

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            filter_img = frame[y1:y2, x1:x2]
            filter_img = FILTERS[current_filter](filter_img)
            frame[y1:y2, x1:x2] = filter_img

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()