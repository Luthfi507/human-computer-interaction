import mediapipe as mp
from mediapipe.tasks.python.vision import (
    drawing_utils, drawing_styles,
    GestureRecognizer, HandLandmarksConnections, GestureRecognizerOptions,
)

model_path = 'models/gesture_recognizer.task'
hand_opt = GestureRecognizerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
HAND_DETECTOR = GestureRecognizer.create_from_options(hand_opt)

def draw_hand_landmarks(frame, hand_landmarks):
    drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        HandLandmarksConnections.HAND_CONNECTIONS,
        drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
    )