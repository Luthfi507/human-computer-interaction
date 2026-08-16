import os
from urllib.request import urlretrieve
from urllib.parse import urlparse
from utils import draw_hand_landmarks

def get_model(url):
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        urlretrieve(url, path)
        print(f"{filename} downloaded")
    
    return path

def landmark_to_pixel(landmark, frame):
    h, w = frame.shape[:2]
    return (int(landmark.x * w), int(landmark.y * h))

def hand_landmark_separator(frame, draw_landmarks: bool, hand_results):
    left_hand = None
    right_hand = None

    for landmarks, handedness in zip(
        hand_results.hand_landmarks, hand_results.handedness
    ):
        if draw_landmarks:
            draw_hand_landmarks(frame, landmarks)
        hand_name = handedness[0].category_name

        if hand_name == "Left":
            left_hand = landmarks
        elif hand_name == "Right":
            right_hand = landmarks

    return left_hand, right_hand