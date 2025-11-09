import os
import cv2
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime
from asl_recognition.mediapipe_tracker import MediapipeTracker

# ============================================================
# Configuration
# ============================================================
SEQ_LEN = 30  # Number of frames per gesture (~1 second)
SAVE_DIR = os.path.join("data", "samples")
os.makedirs(SAVE_DIR, exist_ok=True)

VALID_LABELS = ["hello", "yes", "thank_you", "im_happy", "goodbye"]

# ============================================================
# Main Function
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help=f"Gesture label ({', '.join(VALID_LABELS)})")
    args = parser.parse_args()
    label = args.label.strip().lower()

    if label not in VALID_LABELS:
        print(f"[ERROR] '{label}' is not a valid label. Choose from: {', '.join(VALID_LABELS)}")
        return

    print(f"[INFO] Recording samples for label: '{label}'")
    print("[INFO] Press 'r' to record, 'q' to quit.\n")

    cap = cv2.VideoCapture(0)
    tracker = MediapipeTracker()

    buffer = deque(maxlen=SEQ_LEN)
    sample_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        keypoints, flipped_frame = tracker.extract_keypoints(frame)

        # Display overlay text
        cv2.putText(flipped_frame, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(flipped_frame, "Press 'r' to record, 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("HandsUp - Record Data", flipped_frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            break

        # Start recording
        if key == ord('r'):
            print(f"[INFO] Recording sequence #{sample_count + 1} for '{label}'...")
            buffer.clear()

            for i in range(SEQ_LEN):
                success, frame = cap.read()
                if not success:
                    break

                keypoints, flipped_frame = tracker.extract_keypoints(frame)
                buffer.append(keypoints)

                # Show recording progress
                progress = int((i + 1) / SEQ_LEN * 400)
                cv2.putText(flipped_frame, f"Recording Frame {i+1}/{SEQ_LEN}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(flipped_frame, (10, 60), (10 + progress, 80), (0, 255, 0), -1)

                cv2.imshow("HandsUp - Record Data", flipped_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Save data after collecting SEQ_LEN frames
            if len(buffer) == SEQ_LEN:
                arr = np.stack(list(buffer))  # shape: (SEQ_LEN, features)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{label}_{ts}.npz"
                np.savez_compressed(os.path.join(SAVE_DIR, filename), x=arr, y=label)
                sample_count += 1
                print(f"[✓] Saved {filename}")

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Finished. Recorded {sample_count} samples for label '{label}'.")

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    main()
