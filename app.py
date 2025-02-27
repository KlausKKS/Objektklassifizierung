import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd

# 🔥 Modell & Labels laden
MODEL_PATH = "mobilenet_model.h5"
CSV_FILE = "Anaconda_projects/Objektklassifizierung Mobile Net/training_data/Classes_alle.csv"
IMG_SIZE = (224, 224)

# 📌 Lade Klassen aus CSV
def load_labels(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {str(row["label_id"]): row["class_name"] for _, row in df.iterrows()}
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
        class_name = LABELS.get(str(i), f'{LABELS.get(str(i), "Unbekannt")}')
        confidence = float(predictions[i])
        results.append(f"{i} {class_name}: {confidence:.2%}")
    
    return "\n".join(results)

# 🌍 Streamlit-App
st.title("📸 Objekterkennung mit Webcam")
st.write("Nutze deine Webcam für eine kontinuierliche Erkennung!")

# 📌 Webcam-Unterstützung
camera_image = st.camera_input("📷 Mache ein Bild mit der Webcam")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="📷 Aufgenommenes Bild", use_column_width=True)

    labels = classify_image(image)

    st.write("🔍 Vorhersagen:")
    st.write(labels)
