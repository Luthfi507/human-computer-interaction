import mediapipe as mp
from mediapipe.tasks.python.vision import (
    FaceLandmarkerOptions, FaceLandmarker
)
from .helper import get_model

model_path = get_model('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
face_lm_opt = FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)
FACE_DETECTOR = FaceLandmarker.create_from_options(face_lm_opt)