import cv2
import os
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset"
OUTPUT_PATH = "frames"

FRAMES_PER_VIDEO = 8   # 🔥 reduced for speed + less storage

folders = {
    "fake": os.path.join(DATASET_PATH, "fake")
}

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# PROCESS
# -----------------------------
for label in folders:

    input_path = folders[label]
    output_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_path, exist_ok=True)

    video_files = []

    # 🔥 supports nested folders
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                video_files.append(os.path.join(root, file))

    print(f"\n🚀 Processing {label}: {len(video_files)} videos")

    for video_path in tqdm(video_files):

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            continue

        # 🔥 uniform sampling (fast)
        frame_indices = [
            int(i * total_frames / FRAMES_PER_VIDEO)
            for i in range(FRAMES_PER_VIDEO)
        ]

        # 🔥 clean filename (no Windows issues)
        safe_name = os.path.basename(video_path).split('.')[0]
        safe_name = safe_name.replace(" ", "_").replace("(", "").replace(")", "")

        saved = 0

        for idx in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # 🔥 resize (optional but recommended)
            frame = cv2.resize(frame, (224, 224))

            frame_name = f"{safe_name}_{saved}.jpg"
            save_path = os.path.join(output_path, frame_name)

            # 🔥 compressed save (reduces size massively)
            success = cv2.imwrite(
                save_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            if not success:
                continue  # skip silently

            saved += 1

        cap.release()

print("\n✅ FAST & OPTIMIZED frame extraction completed!")