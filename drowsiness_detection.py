"""
Real-time Drowsiness and Yawning Detection using CNN + OpenCV.
Logs events (Drowsy / Yawning) to CSV and triggers a Windows beep alert.
Dependencies:
    pip install opencv-python numpy tensorflow pillow requests tqdm
"""

import cv2
import numpy as np
import time
import csv
import winsound
from pathlib import Path
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / 'model' / 'eye_model.h5'
CSV_FILE = BASE_DIR / 'drowsiness_log.csv'

EYE_CLOSED_FRAMES = 15
YAWN_THRESHOLD = 25  # Increase for stricter yawn detection
ALARM_FREQ = 1000
ALARM_DURATION = 700

# === CHECK MODEL ===
if not MODEL_PATH.exists():
    print(f"❌ Trained model not found at: {MODEL_PATH}")
    print("Run train_eye_model.py first and ensure model/eye_model.h5 exists.")
    raise SystemExit(1)

# === LOAD MODEL ===
model = load_model(str(MODEL_PATH))

# === CSV SETUP ===
csv_file = open(CSV_FILE, mode='a', newline='')
csv_writer = csv.writer(csv_file)
if csv_file.tell() == 0:
    csv_writer.writerow(['Timestamp', 'Event'])

# === HAAR CASCADES ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# === CAMERA INIT ===
cap = cv2.VideoCapture(0)
eye_closed_counter = 0
ALARM_ON = False

# === FUNCTIONS ===
def predict_eye_state(eye_img):
    """Predicts if the eye is open or closed using CNN."""
    eye_img = cv2.resize(eye_img, (64, 64))
    if eye_img.ndim == 2:
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)
    x = eye_img.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)[0][0]
    return pred  # sigmoid output [0,1]

def beep_alert():
    winsound.Beep(ALARM_FREQ, ALARM_DURATION)

# === MAIN LOOP ===
print("Starting camera. Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        # === Eye Detection ===
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(20, 20))
        if len(eyes) == 0:
            upper = roi_color[0:int(h/2), :]
            hw = upper.shape[1] // 4
            candidates = [upper[:, :2*hw], upper[:, 2*hw:]]
            preds = []
            for cand in candidates:
                try:
                    preds.append(predict_eye_state(cand))
                except Exception:
                    pass
            if preds and np.mean(preds) < 0.5:
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0
        else:
            closed_votes = 0
            total = 0
            for (ex, ey, ew, eh) in eyes:
                eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                try:
                    p = predict_eye_state(eye_img)
                except Exception:
                    p = 1.0
                total += 1
                color = (0, 255, 0)
                if p < 0.5:
                    closed_votes += 1
                    color = (0, 0, 255)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)

            if total > 0 and (closed_votes / total) > 0.5:
                eye_closed_counter += 1
            else:
                eye_closed_counter = 0

        # === Drowsiness Alert ===
        if eye_closed_counter >= EYE_CLOSED_FRAMES and not ALARM_ON:
            ALARM_ON = True
            print("🚨 DROWSINESS ALERT!")
            csv_writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), 'Drowsy'])
            csv_file.flush()
            beep_alert()
        elif eye_closed_counter == 0:
            ALARM_ON = False

        # === Yawning Detection ===
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
        for (mx, my, mw, mh) in mouth_rects:
            mouth_aspect = mh / float(w)
            if mouth_aspect > YAWN_THRESHOLD / 100:  # Normalize by face width
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)
                print("😮 YAWNING DETECTED!")
                csv_writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), 'Yawning'])
                csv_file.flush()
                beep_alert()

        # === Face Rectangle ===
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Live Drowsiness + Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("✅ Session ended and CSV saved.")
