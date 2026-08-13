import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceDetectorOptions, FaceDetector,
    FaceLandmarkerOptions, FaceLandmarker
)

from tools import FILTERS, HandController
from utils import HAND_DETECTOR, draw_hand_landmarks

hand_controller = HandController()
opt = FaceDetectorOptions(base_options=mp.tasks.BaseOptions(model_asset_path="models/blaze_face_full_range.tflite"))
detector = FaceDetector.create_from_options(opt)

face_opt = FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path="models/face_landmarker.task"))
face_landmarker = FaceLandmarker.create_from_options(face_opt)

def process_face(frame, face_results, current_filter):
    for result in face_results.detections:
        bboxes = result.bounding_box

        x1, y1 = bboxes.origin_x, bboxes.origin_y
        x2, y2 = bboxes.origin_x + bboxes.width, bboxes.origin_y + bboxes.height

        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = FILTERS[current_filter](roi)

    return frame

def process_face_landmarks(frame, face_landmarks, current_filter):
    frame_h, frame_w = frame.shape[:2]
    points = np.array([
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in face_landmarks
    ], np.int32)

    hull = cv2.convexHull(points)
    x, y, bw, bh = cv2.boundingRect(hull)

    # guard against empty / out-of-bounds regions
    if bw == 0 or bh == 0:
        return frame

    roi = frame[y:y+bh, x:x+bw]
    if roi.size == 0:
        return frame

    filtered = FILTERS[current_filter](roi)

    local_hull = hull - np.array([x, y])
    
    mask = np.zeros((bh, bw), np.uint8)
    cv2.fillConvexPoly(mask, local_hull, 255)

    # ensure mask shape matches roi/filter shapes
    if mask.shape[:2] != roi.shape[:2] or mask.shape[:2] != filtered.shape[:2]:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    roi[mask == 255] = filtered[mask == 255]
    return frame

def process_pinch(frame, hand_results):
    pinching = False

    for landmarks in hand_results.hand_landmarks:
        draw_hand_landmarks(frame, landmarks)
        pinching = hand_controller.update(landmarks)

    return pinching

def process_polygon(frame, hand_results, current_filter):
    points = hand_controller.get_poly_point(
        hand_results,
        frame.shape
    )

    if points is None:
        return frame

    x, y, w, h = cv2.boundingRect(points)

    roi = frame[y:y+h, x:x+w]
    filtered_roi = FILTERS[current_filter](roi.copy())

    local_points = points - np.array([x, y])
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [local_points], 255)

    mask_3ch = cv2.cvtColor(
        mask, cv2.COLOR_GRAY2BGR
    ).astype(np.float32) / 255.0

    blended = (
        filtered_roi * mask_3ch + roi * (1 - mask_3ch)
    ).astype(np.uint8)

    frame[y:y+h, x:x+w] = blended

    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera detected")

    current_filter = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
        face_results = face_landmarker.detect(mp_image)
        hand_results = HAND_DETECTOR.detect(mp_image)

        pinching = process_pinch(frame, hand_results)
        if pinching:
            current_filter = (
                current_filter + 1
            ) % len(FILTERS)

        # frame = process_face(frame, face_results, current_filter - 1)
        frame = process_polygon(frame, hand_results, current_filter)
        # for face_landmarks in face_results.face_landmarks:
        #     frame = process_face_landmarks(frame, face_landmarks, current_filter)

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()