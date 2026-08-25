import cv2
import mediapipe as mp

from engines import (
    RectanglePipeline, 
    PolyPipeline, 
    CirclePipeline, 
    SpotlightPipeline, 
    SelfieSegmentation,
    SpotlightBackground,
    Inivisible
)
from utils import HAND_DETECTOR, SEGMENTER, FACE_DETECTOR
from tools import FaceExpression

MODES = {
    'spotbg': SpotlightBackground,
    'splotlight': SpotlightPipeline, 
    'circle': CirclePipeline, 
    'poly': PolyPipeline, 
    'rectangle': RectanglePipeline, 
    'selfie': SelfieSegmentation,
}
mode_name = list(MODES.keys())
expression = FaceExpression(0.5)

class Effect:
    def __init__(self, draw: bool = False, camera=0):
        self.camera = camera
        self.draw = draw
        self.current_mode = 0
        self.cap = None
        self._mode_name = mode_name[self.current_mode]
        self.invisible = Inivisible()

    def get_camera(self):
        self.cap = cv2.VideoCapture(self.camera)
        if not self.cap.isOpened():
            raise RuntimeError("No camera detected")

    def update_mode(self):
        self.current_mode = (self.current_mode + 1) % len(mode_name)
        self._mode_name = mode_name[self.current_mode]
        return MODES[self._mode_name](self.draw)

    def run(self):
        self.get_camera()
        processor = MODES[self._mode_name](self.draw)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

            hand_results = HAND_DETECTOR.recognize(mp_image)
            face_results = FACE_DETECTOR.detect(mp_image)
            seg_results = SEGMENTER.segment(mp_image)

            frame = processor.process(frame=frame, hand_results=hand_results, seg_results=seg_results)
            frame = self.invisible.process(frame, hand_results=hand_results, seg_results=seg_results)
            frame = cv2.flip(frame, 1)

            face_blendshapes = face_results.face_blendshapes
            if face_blendshapes:
                trigger = expression.update(
                    face_blendshapes[0], 
                    [
                        ("is_brow_up", {"threshold": 0.01})
                    ]
                )

                if trigger:
                    print(self._mode_name)
                    processor = self.update_mode()

            cv2.imshow(
                "camera",
                cv2.resize(frame, None, fx=1.5, fy=1.5)
                # frame
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipe = Effect(True)
    pipe.run()