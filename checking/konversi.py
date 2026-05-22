import os
import cv2

base_path = "D:/post_kuliah/assessment/project-2/dataset"
output_path = "D:/post_kuliah/assessment/project-2/classification_dataset"

# mapping dari data.yaml kamu
class_names = ["depan", "miring_kanan", "miring_kiri", "nunduk"]

splits = ["train", "valid", "test"]

for split in splits:
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")

    for file in os.listdir(img_dir):
        if not file.endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path) as f:
            for i, line in enumerate(f):
                cls, x, y, bw, bh = map(float, line.split())

                # convert YOLO → pixel
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                crop = img[y1:y2, x1:x2]

                class_name = class_names[int(cls)]
                save_dir = os.path.join(output_path, split, class_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{file[:-4]}_{i}.jpg")
                cv2.imwrite(save_path, crop)

print("Selesai konversi 🚀")