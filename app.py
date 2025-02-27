import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import requests
from io import BytesIO

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
    for i in top_2:
        class_name = LABELS.get(i, "Unbekannt")
        confidence = float(predictions[i])
        results.append(f"{class_name}: {confidence:.2%}")

    return "\n".join(results), image

# 🌍 Streamlit-App
st.title("📹 Echtzeit-Objekterkennung mit Flask-Livestream")
st.write("Starte den Flask-Server, um den Livestream zu empfangen.")

# 📌 Livestream von Flask abrufen
stream_url = "http://localhost:5000/video"

if st.button("🎥 Starte Livestream"):
    st.write("📡 Verbinde mit dem Livestream...")
    while True:
        try:
            response = requests.get(stream_url, stream=True)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                labels, output_image = classify_image(image)
                st.image(output_image, caption=f"🔍 {labels}", use_column_width=True)
            else:
                st.error("❌ Kein Bild empfangen, prüfe den Flask-Server!")
        except Exception as e:
            st.error(f"Fehler beim Abrufen des Livestreams: {e}")
            break
