import mediapipe as mp
from mediapipe.tasks.python.vision import (
    ImageSegmenter, ImageSegmenterOptions
)
from .helper import get_model

seg_url = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite'
seg_path = get_model(seg_url)

seg_opt = ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=seg_path),
    output_category_mask=True
)
SEGMENTER = ImageSegmenter.create_from_options(seg_opt)