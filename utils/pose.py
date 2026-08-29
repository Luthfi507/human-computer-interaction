import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmarkerOptions, PoseLandmarker
)
from .helper import get_model

model_path = get_model('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task')
pose_opt = PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    output_segmentation_masks=True
)
POSE_DETECTOR = PoseLandmarker.create_from_options(pose_opt)