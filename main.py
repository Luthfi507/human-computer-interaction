import cv2
import mediapipe as mp

from engines import MODES
from utils import HAND_DETECTOR

draw = True

def main():
    processor = MODES[2](draw)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No camera detected")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)
        hand_results = HAND_DETECTOR.detect(mp_image)

        frame = processor.process(frame, hand_results)
        print(processor.current_filter)

        cv2.imshow(
            "Camera",
            # cv2.resize(frame, None, fx=2, fy=2)
            frame
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()