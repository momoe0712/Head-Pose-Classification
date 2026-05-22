import cv2
import numpy as np
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("mobilenetv3_models.keras")

class_names = ["depan", "miring_kanan", "miring_kiri", "nunduk"]
IMG_SIZE = 224

# =========================
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =========================
    # DETECT FACE
    # =========================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # =========================
        # PREPROCESS
        # =========================
        img = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.mobilenet_v3.preprocess_input(img)

        # =========================
        # PREDICT
        # =========================
        preds = model.predict(img, verbose=0)
        class_idx = np.argmax(preds)
        confidence = preds[0][class_idx]

        label = f"{class_names[class_idx]} ({confidence:.2f})"

        # =========================
        # DRAW BOX (DARK BLUE)
        # =========================
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (139, 0, 0),  # dark blue (BGR)
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (139, 0, 0),
            2
        )

    cv2.imshow("Head Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()