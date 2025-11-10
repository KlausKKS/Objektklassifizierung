import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tempfile
import os
import time

# === Setup ===
MODEL_PATH = "mobilenet_model_v3.keras"
CLASSES_CSV = "training_data/Classes_alle.csv"
RULES_CSV = "training_data/Abmessungen.csv"

IMG_SIZE = (224, 224)
kalibrierfaktor = {"10x": 0.728, "20x": 0.363, "40x": 0.181, "63x": 0.115}

# === Cache laden ===
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    labels = pd.read_csv(CLASSES_CSV, sep=";")
    rules = pd.read_csv(RULES_CSV, sep=";")
    label_dict = dict(zip(labels["label_id"].astype(int), labels["class_name"]))
    rule_dict = {row["klasse"]: row for _, row in rules.iterrows()}
    return model, label_dict, rule_dict

model, LABELS, RULES = load_resources()

# === Funktionen ===
def preprocess_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def classify_top2(frame):
    preds = model.predict(preprocess_frame(frame), verbose=0)[0]
    top = preds.argsort()[-2:][::-1]
    return [(LABELS.get(i, f"ID {i}"), float(preds[i])) for i in top]

# === Streamlit UI ===
st.set_page_config(page_title="Objekterkennung Desmids", layout="wide")

st.title("🔬 Objekterkennung & Längenmessung")
st.markdown("### Funktionen: Bild aufnehmen oder hochladen")

# Kamera oder Upload
mode = st.radio("Quelle wählen", ["Kamera", "Datei-Upload"])
img = None

if mode == "Kamera":
    img_data = st.camera_input("Foto aufnehmen")
    if img_data:
        file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
elif mode == "Datei-Upload":
    uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

# Wenn Bild vorhanden:
if img is not None:
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Originalbild", use_column_width=True)

    top2 = classify_top2(img)
    st.subheader("Top-2 Klassifikation:")
    for label, conf in top2:
        st.write(f"**{label}** – {conf:.1%}")

    # Längenmessung (zwei Klicks auf das Bild)
    st.markdown("---")
    st.markdown("### Längenmessung")
    st.info("📏 In dieser Web-Version ist die Messung noch manuell zu ergänzen.")
    st.caption("In der Desktop-Version erfolgt sie über Mausklicks auf zwei Punkte.")

    # Speicherung
    if st.button("📸 Bild speichern"):
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = f"bilder/{ts}.jpg"
        os.makedirs("bilder", exist_ok=True)
        cv2.imwrite(out_path, img)
        st.success(f"Gespeichert als {out_path}")

else:
    st.warning("Bitte ein Bild aufnehmen oder hochladen.")
