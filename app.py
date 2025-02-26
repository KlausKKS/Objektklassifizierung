import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import time

# 🔥 Modell & Labels laden
MODEL_PATH = "mobilenet_model.h5"
CSV_FILE = "training_data/Classes_alle.csv"
IMG_SIZE = (224, 224)

# 📌 Lade Klassen aus CSV
def load_labels(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {int(row["label_id"]): row["class_name"] for _, row in df.iterrows()}
    return {}

LABELS = load_labels(CSV_FILE)
model = tf.keras.models.load_model(MODEL_PATH)

# 📌 Bild vorbereiten
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# 📌 Bildklassifikation
def classify_image(image):
    img_tensor = preprocess_image(image)
    predictions = model.predict(img_tensor)[0]
    top_2 = np.argsort(predictions)[-2:][::-1]

    results = []
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    y_offset = 50
    for i in top_2:
        class_name = LABELS.get(i, f"Klasse {i}")  # 🔄 Jetzt mit echten Klassennamen
        confidence = float(predictions[i])
        label_text = f"{class_name}: {confidence:.2%}"
        results.append(label_text)

        # 🔥 Text ins Bild zeichnen
        cv2.putText(frame_bgr, label_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        y_offset += 50  

    return "\n".join(results), Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

# 🌍 Streamlit-App
st.title("📸 Fast-Live Objekterkennung mit Webcam")
st.write("Nutze deine Webcam für eine kontinuierliche Erkennung!")

# 📌 Webcam-Unterstützung
camera_image = st.camera_input("📷 Mache ein Bild mit der Webcam")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="📷 Aufgenommenes Bild", use_column_width=True)

    frame_placeholder = st.empty()

    if st.button("🎥 Starte Fast-Live Verarbeitung"):
        for _ in range(20):  # 🔄 Simuliert Live-Update für 20 Durchläufe
            labels, output_image = classify_image(image)
            frame_placeholder.image(output_image, caption=f"🔍 {labels}", use_column_width=True)
            time.sleep(0.5)  # 🔄 Aktualisierung alle 0,5 Sekunden
