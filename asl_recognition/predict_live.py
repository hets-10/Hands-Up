import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import os
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
from integration.elevenlabs_voice import speak_text
import time

# Import custom modules
from asl_recognition.mediapipe_tracker import MediapipeTracker
from integration.neuralseek_integration import improve_sentence

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.path.join("models", "asl_model.h5")
LABEL_PATH = os.path.join("models", "labels.txt")
SEQ_LEN = 30  # must match training
CONF_THRESHOLD = 0.8  # only speak if model is confident

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize Mediapipe tracker
tracker = MediapipeTracker()
sequence = deque(maxlen=SEQ_LEN)
prev_label = None
last_spoken_time = 0
COOLDOWN = 1.5  # seconds between voice outputs

# ============================================================
# Start webcam
# ============================================================
cap = cv2.VideoCapture(0)
print("[INFO] Starting live ASL recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    keypoints, annotated_frame = tracker.extract_keypoints(frame)

    if keypoints is not None:
        sequence.append(np.array(keypoints).flatten())

        if len(sequence) == SEQ_LEN:
            x_input = np.expand_dims(sequence, axis=0)  # shape (1, 30, 126)
            preds = model.predict(x_input, verbose=0)[0]
            pred_label = labels[np.argmax(preds)]
            confidence = np.max(preds)

            if confidence > CONF_THRESHOLD:
                # Convert label -> natural English sentence
                sentence = improve_sentence(pred_label)

                # Display on screen
                cv2.putText(annotated_frame, f"{sentence} ({confidence*100:.1f}%)",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                # Speak only when new prediction and cooldown passed
                if pred_label != prev_label and (time.time() - last_spoken_time) > COOLDOWN:
                    print(f"[VOICE] Speaking: {sentence}")
                    speak_text(sentence)
                    prev_label = pred_label
                    last_spoken_time = time.time()
            else:
                cv2.putText(annotated_frame, "Uncertain gesture",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_frame, "No hands detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("HandsUp - Live ASL Prediction", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
tracker.close()
cv2.destroyAllWindows()
print("[INFO] Stream ended.")