import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small

# === Streamlit-Setup ===
st.set_page_config(page_title="Objekterkennung Desmids", layout="wide")
st.title("🔬 Objekterkennung & Klassifikation")
st.caption("MobileNetV3 – Top-2-Ergebnisse mit Kamera oder Datei-Upload")

# === Dateipfade ===
MODEL_PATH = "mobilenet_model_v3_fixed.keras"
CLASSES_CSV = "training_data/Classes_alle.csv"
RULES_CSV = "training_data/Abmessungen.csv"
IMG_SIZE = (224, 224)

# === Ressourcen laden (mit Cache) ===
@st.cache_resource
def load_resources():
    # Versuch: zuerst Large, dann Small
    try:
        model = load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"Functional": MobileNetV3Large}
        )
        model_type = "MobileNetV3Large"
    except Exception as e1:
        print("⚠️ Large nicht erkannt, versuche Small...")
        model = load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"Functional": MobileNetV3Small}
        )
        model_type = "MobileNetV3Small"

    df_labels = pd.read_csv(CLASSES_CSV, sep=";")
    df_rules = pd.read_csv(RULES_CSV, sep=";")
    labels = dict(zip(df_labels["label_id"].astype(int), df_labels["class_name"]))
    rules = {r["klasse"]: r for _, r in df_rules.iterrows()}

    return model, labels, rules, model_type

# === Modell und CSV-Dateien laden ===
model, LABELS, RULES, MODEL_TYPE = load_resources()
st.sidebar.success(f"✅ Modell geladen: {MODEL_TYPE}")

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

# === Benutzeroberfläche ===
modus = st.radio("📸 Quelle wählen", ["Kamera", "Datei-Upload"])
frame = None

if modus == "Kamera":
    cam_data = st.camera_input("Foto aufnehmen")
    if cam_data:
        file_bytes = np.asarray(bytearray(cam_data.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
elif modus == "Datei-Upload":
    upload = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
    if upload:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

# === Klassifikation ===
if frame is not None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Aufgenommenes Bild", use_column_width=True)

    top2 = classify_top2(frame)
    st.subheader("Top-2 Klassifikation")
    for label, conf in top2:
        st.write(f"**{label}** – {conf:.1%}")

    # === Bild speichern ===
    if st.button("📸 Bild speichern"):
        ts = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs("bilder", exist_ok=True)
        out_path = f"bilder/{ts}.jpg"
        cv2.imwrite(out_path, frame)
        st.success(f"Gespeichert als {out_path}")

else:
    st.warning("Bitte ein Bild aufnehmen oder hochladen.")

# === Footer ===
st.markdown("---")
st.caption("© 2025 – Objekterkennung Desmids • Entwickelt für Streamlit Cloud & MobileNetV3")
