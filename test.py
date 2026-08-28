import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import mediapipe as mp
from utils import HAND_DETECTOR
from engines.render import HandObjectRenderer

cap = cv2.VideoCapture(0)
if not cap:
    raise RuntimeError("No camera")

glb_renderer = HandObjectRenderer()
angle = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

        hand_results = HAND_DETECTOR.recognize(mp_image)

        if hand_results.hand_landmarks:
            angle += 0.2
            hand = hand_results.hand_landmarks[0]

            glb_renderer.update_pose(hand, angle)
            frame, _ = glb_renderer.render_overlay(frame)

        frame = cv2.flip(frame, 1)
        cv2.imshow("Webcam + GLB", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    glb_renderer.close()
    cap.release()
    cv2.destroyAllWindows()
