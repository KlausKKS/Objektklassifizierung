import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import time
from PIL import Image

# 🔥 Modell & Labels laden
MODEL_PATH = "mobilenet_model.h5"
CSV_FILE = "training_data/Classes_alle.csv"
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.001

# 📌 Lade Klassen aus CSV
def load_labels_from_csv(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {int(row['label_id']): row['class_name'] for _, row in df.iterrows()}
    return {}

LABELS = load_labels_from_csv(CSV_FILE)
num_classes = len(LABELS)

# 📌 Modell laden oder neu erstellen
def get_model(num_classes):
    if num_classes == 0:
        st.error("❌ Keine Klassen gefunden. Modell wird nicht geladen.")
        return None
    
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = get_model(num_classes)

if model and os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)
    st.success("✅ Modell geladen!")
else:
    st.warning("❌ Kein trainiertes Modell gefunden!")

# 📌 Bildvorbereitung
def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
    frame_array = np.array(frame_resized) / 255.0
    return np.expand_dims(frame_array, axis=0)

# 📌 Klassifikation
def classify_image(model, frame):
    img_tensor = preprocess_frame(frame)
    predictions = model.predict(img_tensor, verbose=0)[0]
    top_2 = np.argsort(predictions)[-2:][::-1]
    return [(LABELS.get(i, f"{i} Unbekannt"), predictions[i]) for i in top_2]

# 📌 Streamlit Webcam-Livestream
st.title("📹 Echtzeit-Objekterkennung mit Streamlit & OpenCV")
st.write("Die Kamera wird genutzt, um kontinuierlich Objekte zu erkennen.")

# Kamera-Stream
FRAME_WINDOW = st.image([])  # Streamlit-Container für das Kamerabild
cap = cv2.VideoCapture(0)  # OpenCV-Kamera starten

if not cap.isOpened():
    st.error("❌ Fehler: Kamera nicht verfügbar!")
else:
    st.success("🎥 Kamera läuft...")

# 📌 Live-Kamera mit Erkennung
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Fehler beim Lesen des Kamerabilds!")
        break

    # Klassifikation
    predictions = classify_image(model, frame)
    text_lines = [f"{label}: {prob:.2%}" for label, prob in predictions]

    # Text ins Bild einfügen
    y_offset = 50
    for line in text_lines:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        y_offset += 50

    # Konvertiere Bild für Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame, channels="RGB")

cap.release()
cv2.destroyAllWindows()
