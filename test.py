import cv2
import numpy as np
import mediapipe as mp
from utils import FACE_DETECTOR
from tools import FaceExpression

expression = FaceExpression(0.5)
cap = cv2.VideoCapture(0)
if not cap:
    raise RuntimeError("No camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, rgb)

    face_result = FACE_DETECTOR.detect(mp_image)
    category = "normal"

    face_blendshapes = face_result.face_blendshapes
    if face_blendshapes:
        is_brow_up = expression.update(face_blendshapes[0], [("is_brow_up", {"threshold": 0.01})])

        if is_brow_up:
            category = "brow up"
    else:
        category = "no face detected"

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, category, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()