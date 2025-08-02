
import streamlit as st
import numpy as np
from PIL import Image
import time
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# === Konfiguration ===
MODEL_PATH = "mobilenet_model.h5"
CLASSES_CSV = "Classes_alle.csv"
RULES_CSV = "Abmessungen.csv"
IMG_SIZE = (224, 224)

# === Laden ===
def load_labels(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    return dict(zip(df["label_id"].astype(int), df["class_name"]))

def load_rules(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    return {row["klasse"]: row for _, row in df.iterrows()}

LABELS = load_labels(CLASSES_CSV)
RULES = load_rules(RULES_CSV)
model = load_model(MODEL_PATH)

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def classify_top2(frame):
    preds = model.predict(preprocess(frame), verbose=0)[0]
    top2 = preds.argsort()[-2:][::-1]
    return [(LABELS[i], preds[i]) for i in top2]

# === Streamlit UI ===
st.title("📷 Objekterkennung mit Kameraaufnahme")

img_data = st.camera_input("📸 Bild aufnehmen")

if img_data is not None:
    image = Image.open(img_data)
    frame = np.array(image.convert("RGB"))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        top2 = classify_top2(frame)
        for label, conf in top2:
            regel = RULES.get(label)
            farbe = "✅" if regel is None else "⚠️"
            st.markdown(f"{farbe} **{label}**: {conf:.1%}")
    except Exception as e:
        st.error(f"Klassifikation fehlgeschlagen: {e}")

    st.image(image, caption="Aufgenommenes Bild", use_column_width=True)
