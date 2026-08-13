import cv2
import mediapipe as mp

from engines import Pipeline
from utils import HAND_DETECTOR

effect = Pipeline()

def main():
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

        frame = effect.polly_process(frame, hand_results)

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()