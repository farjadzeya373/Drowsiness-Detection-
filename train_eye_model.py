"""
train_eye_model.py
Train a light CNN to classify open vs closed eyes using images in your dataset.

USAGE:
    python train_eye_model.py

Trains and saves model/model/eye_model.h5
"""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ===================== CONFIG =====================
# Use your absolute dataset path here
DATA_DIR = Path(r"C:\Users\Lenovo\Downloads\drowsiness_detection_windows\dataset\eyes")
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / 'model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 12

# ===================== MODEL =====================
def build_model(input_shape=(64,64,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ===================== TRAINING =====================
def main():
    if not DATA_DIR.exists():
        print("❌ Dataset directory not found:", DATA_DIR)
        print("Please ensure images are under:")
        print("  -", DATA_DIR / 'open')
        print("  -", DATA_DIR / 'closed')
        return

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR / 'eye_model.h5'), save_best_only=True, monitor='val_accuracy', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
    print('✅ Model saved to', MODEL_DIR / 'eye_model.h5')

if __name__ == '__main__':
    main()
