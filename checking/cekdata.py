import os
from collections import Counter
import yaml

# folder dataset
base_path = "D:/post_kuliah/assessment/project-2/dataset"

# folder label
dirs = ["train/labels", "valid/labels", "test/labels"]

# baca data.yaml
with open(os.path.join(base_path, "data.yaml")) as f:
    data = yaml.safe_load(f)

names = data["names"]

# counter
class_counts = Counter()

for d in dirs:
    full_path = os.path.join(base_path, d)
    
    if not os.path.exists(full_path):
        print(f"Folder tidak ditemukan: {full_path}")
        continue

    for file in os.listdir(full_path):
        if file.endswith(".txt"):
            with open(os.path.join(full_path, file)) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

# print hasil
print("\nJumlah objek per kelas:\n")

for cls_id, count in sorted(class_counts.items()):
    class_name = names[cls_id] if isinstance(names, list) else names[str(cls_id)]
    print(f"{cls_id} ({class_name}): {count}")