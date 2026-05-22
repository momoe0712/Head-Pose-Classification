# Head Pose Classification 

Real-time head pose detection using **MobileNetV3Small** deployed on Android via TensorFlow Lite.

---

## Overview

This project classifies head orientation in real-time using the front camera of an Android device. The model was trained with TensorFlow/Keras and deployed as a TFLite model inside an Android app built with Jetpack Compose and CameraX.

| Item | Detail |
|---|---|
| Model | MobileNetV3Small (transfer learning) |
| Framework (Training) | TensorFlow / Keras |
| Framework (Mobile) | TensorFlow Lite |
| Platform | Android (Jetpack Compose + CameraX) |
| Input Size | 224 × 224 px |
| Classes | 4 |
| Test Accuracy | **98%** |

---

## Classes

| Label | Description |
|---|---|
| `depan` | Head facing forward |
| `miring_kanan` | Head tilted to the right |
| `miring_kiri` | Head tilted to the left |
| `nunduk` | Head looking down |

---

## Project Structure

```
├── android/
│   └── app/src/main/
│       ├── assets/
│       │   └── model.tflite          # Converted TFLite model
│       └── java/com/example/myapplication/
│           └── MainActivity.kt       # Main activity with CameraX + inference
│
├── training/
│   └── trainten.ipynb                # Training notebook (TensorFlow/Keras)
│
└── README.md
```

---

## Dataset

```
Train : 2609 images  (4 classes)
Valid : 745  images  (4 classes)
Test  : 373  images  (4 classes)
```

Images were organized in subdirectories by class and loaded using `tf.keras.utils.image_dataset_from_directory`.

---

## Training

The model was trained using **transfer learning** on top of MobileNetV3Small pretrained with ImageNet weights.

```python
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Preprocessing is built into the model graph
x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(4, activation="softmax")(x)
```

Key training config:

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 32 |
| Base model frozen | Yes |

---

## Model Conversion (Keras → TFLite)

```python
model = tf.keras.models.load_model("mobilenetv3_models.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

> **Note:** `preprocess_input` is baked into the model graph, so the Android app sends **raw pixel values (0–255)** directly to the model without any manual normalization.

---

## 📱Android Implementation

### Key Components

- **CameraX** — front camera live feed with `ImageAnalysis`
- **TFLite Interpreter** — runs inference on each frame
- **Jetpack Compose** — UI layer

---

## Getting Started

### Requirements

- Android Studio Hedgehog or later
- Android device / emulator with front camera
- Min SDK: 24

[Watch Demo](https://drive.google.com/file/d/11IBZu_B3DmxERsnHpv-J2ccTkwvuS0q_/view?usp=sharing)

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **98%** |
| Inference Target | Real-time (front camera) |
| Detected Classes | depan, miring_kanan, miring_kiri, nunduk |

---

## Known Issues & Notes

- We believe the dataset that we used in this project are still lacking in quantity. With more quantity in dataset a better quality of detection can be achieve.
- **Do not normalize pixel values in Android** — normalization is handled internally by the model via `preprocess_input`. Normalizing twice will break predictions.
- Camera rotation must be handled explicitly using `imageInfo.rotationDegrees` to ensure correct orientation-sensitive predictions.
- For best accuracy, ensure the face is well-lit and centered in frame.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model Training | TensorFlow 2.x, Keras |
| Base Model | MobileNetV3Small (ImageNet) |
| Mobile Runtime | TensorFlow Lite |
| Android UI | Jetpack Compose |
| Camera | CameraX |
| Language | Kotlin |
