import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# -----------------------------
# PATHS (FIXED according to your folder)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "metrics", "best_model.h5")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# PARAMETERS (TUNE THESE)
# -----------------------------
IMG_SIZE = 224
FRAME_SKIP = 10

# 🔥 VERY IMPORTANT FIX
FAKE_THRESHOLD = 0.75   # increased from 0.5 → reduces false fake
REAL_THRESHOLD = 0.25   # strong real confidence

MIN_CONFIDENCE = 0.6    # ignore weak predictions

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# -----------------------------
# VIDEO PREDICTION
# -----------------------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fake_scores = []
    real_scores = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            processed = preprocess(frame)
            prob = model.predict(processed, verbose=0)[0][0]

            confidence = abs(prob - 0.5) * 2  # 0 → uncertain, 1 → strong

            # ✅ Ignore weak predictions
            if confidence < MIN_CONFIDENCE:
                frame_count += 1
                continue

            if prob > FAKE_THRESHOLD:
                fake_scores.append(prob)

            elif prob < REAL_THRESHOLD:
                real_scores.append(1 - prob)

        frame_count += 1

    cap.release()

    # -----------------------------
    # DECISION LOGIC (SMART VOTING)
    # -----------------------------
    if len(fake_scores) == 0 and len(real_scores) == 0:
        return "UNCERTAIN", 0

    avg_fake = np.mean(fake_scores) if fake_scores else 0
    avg_real = np.mean(real_scores) if real_scores else 0

    print(f"\nDEBUG → Fake score: {avg_fake:.3f}, Real score: {avg_real:.3f}")

    # 🔥 FINAL DECISION
    if avg_fake > avg_real:
        return "FAKE", avg_fake
    else:
        return "REAL", avg_real


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    video_path = input("003.mp4 ")

    result, confidence = predict_video(video_path)

    print(f"\n🎯 Prediction: {result}")
    print(f"🔥 Confidence: {confidence:.2f}")