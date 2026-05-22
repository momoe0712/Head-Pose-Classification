import cv2
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================
# model = tf.keras.models.load_model("mobilenetv3_models.keras")
model = tf.keras.models.load_model("mobilenetv3_models.h5")

# =========================
# CLASS NAMES (sesuaikan urutan!)
# =========================
class_names = ["depan", "miring_kanan", "miring_kiri", "nunduk"]

IMG_SIZE = 224

# =========================
# OPEN CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # PREPROCESS
    # =========================
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    # IMPORTANT: same preprocessing as training
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)

    # =========================
    # PREDICT
    # =========================
    preds = model.predict(img, verbose=0)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    label = f"{class_names[class_idx]} ({confidence:.2f})"

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Head Movement Detection", frame)

    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 