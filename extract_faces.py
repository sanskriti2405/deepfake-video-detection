import os
import cv2
from tqdm import tqdm
from mtcnn import MTCNN

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAME_PATH = os.path.join(BASE_DIR, "frames")
FACE_PATH = os.path.join(BASE_DIR, "faces")

os.makedirs(FACE_PATH, exist_ok=True)

# -----------------------------
# INIT DETECTOR
# -----------------------------
detector = MTCNN()

# -----------------------------
# SETTINGS (IMPORTANT)
# -----------------------------
MAX_IMAGES_PER_CLASS = 5000  # reduce if slow

# -----------------------------
# FACE EXTRACTION FUNCTION
# -----------------------------
def extract_face(image):
    try:
        results = detector.detect_faces(image)
    except:
        return None

    if len(results) == 0:
        return None

    x, y, w, h = results[0]['box']

    # Fix negative values
    x, y = max(0, x), max(0, y)

    face = image[y:y+h, x:x+w]

    if face.size == 0:
        return None

    return cv2.resize(face, (224, 224))

# -----------------------------
# MAIN LOOP
# -----------------------------
for label in ["real", "fake"]:

    input_path = os.path.join(FRAME_PATH, label)
    output_path = os.path.join(FACE_PATH, label)

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"❌ Folder not found: {input_path}")
        continue

    images = [f for f in os.listdir(input_path) if f.endswith(".jpg")]

    print(f"\n🚀 Processing {label}: {len(images)} images")

    count = 0

    for img_name in tqdm(images):

        if count >= MAX_IMAGES_PER_CLASS:
            break

        img_path = os.path.join(input_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        face = extract_face(image)

        if face is None:
            continue

        save_path = os.path.join(output_path, img_name)

        try:
            cv2.imwrite(save_path, face)
            count += 1
        except:
            continue

print("\n✅ DONE: Faces extracted successfully!")