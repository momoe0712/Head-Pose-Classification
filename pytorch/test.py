import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# ======================
# LOAD MODEL
# ======================
class_names = ["depan", "miring_kanan", "miring_kiri", "nunduk"]

model = models.mobilenet_v3_small()
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))

model.load_state_dict(torch.load("best_mobilenetv3_model.pth", map_location="cpu"))
model.eval()

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()

    label = class_names[pred]

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Realtime", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()