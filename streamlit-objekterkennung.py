
import os
os.chdir(os.path.dirname(__file__))
import streamlit as st
import cv2
import numpy as np
import time
import os
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# === Streamlit-Konfiguration ===
st.set_page_config(layout="wide")

# === Konfiguration ===
MODEL_PATH = "mobilenet_model.h5"
CLASSES_CSV = "training_data/Classes_alle.csv"
RULES_CSV = "training_data/Abmessungen.csv"
IMG_SIZE = (224, 224)
BILD_SAVE_DIR = "bilder"
KORREKTUR_DIR = "korrigierte_daten"
NEUE_KLASSEN_DIR = "neue_klassen"

os.makedirs(BILD_SAVE_DIR, exist_ok=True)
os.makedirs(KORREKTUR_DIR, exist_ok=True)
os.makedirs(NEUE_KLASSEN_DIR, exist_ok=True)

kalibrierfaktor = {
    "10x": 0.728,
    "20x": 0.363,
    "40x": 0.181,
    "63x": 0.115
}

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

# === Klassifikation ===
def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def classify_top2(frame):
    preds = model.predict(preprocess(frame), verbose=0)[0]
    top2 = preds.argsort()[-2:][::-1]
    return [(LABELS[i], preds[i]) for i in top2]

# === Streamlit UI ===
st.title("🔬 Live Objekterkennung & Messung")

objektiv = st.selectbox("🔍 Objektiv wählen", list(kalibrierfaktor.keys()), index=0)
zoom = st.slider("🔍 Digital-Zoom", 1.0, 3.0, 1.0, 0.1)

capture = st.checkbox("📸 HD-Bild speichern")
correct = st.checkbox("✏️ Klassifikation korrigieren")
new_class = st.checkbox("➕ Neue Klasse hinzufügen")

frame_placeholder = st.empty()
results_placeholder = st.empty()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Kamera nicht verfügbar")
    st.stop()

def digital_zoom(frame, zoom_factor):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    x1 = max(center_x - new_w // 2, 0)
    y1 = max(center_y - new_h // 2, 0)
    x2 = x1 + new_w
    y2 = y1 + new_h
    zoomed = frame[y1:y2, x1:x2]
    return cv2.resize(zoomed, (w, h))

with st.spinner("📷 Starte Kamera..."):
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Kein Frame erhalten")
        st.stop()

    frame_display = digital_zoom(frame, zoom)
    annotated = frame_display.copy()

    # Klassifikation
    try:
        top2 = classify_top2(frame)
        lines = []
        for label, conf in top2:
            regel = RULES.get(label)
            farbe = "✅" if regel is None else "⚠️"
            lines.append(f"{farbe} **{label}**: {conf:.1%}")
        results_placeholder.markdown("\n".join(lines))
    except Exception as e:
        results_placeholder.error(f"Klassifikation fehlgeschlagen: {e}")

    # Anzeige
    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_placeholder.image(frame_pil, channels="RGB")

    # Bild speichern
    if capture:
        ts = time.strftime("%Y%m%d-%H%M%S")
        path_hd = os.path.join(BILD_SAVE_DIR, f"{ts}_HD.jpg")
        cv2.imwrite(path_hd, frame)
        path_ann = os.path.join(BILD_SAVE_DIR, f"{ts}_annotiert.jpg")
        cv2.imwrite(path_ann, annotated)
        st.success(f"📸 Bild gespeichert: {path_hd}\n📝 Annotiert: {path_ann}")

    # Klassifikation korrigieren
    if correct:
        klassen_liste = sorted(set(LABELS.values()))
        auswahl = st.selectbox("Neue Klasse wählen", klassen_liste)
        if st.button("💾 Speichern unter korrigierte Klasse"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(KORREKTUR_DIR, f"{auswahl}_{ts}.jpg")
            cv2.imwrite(path, frame)
            st.success(f"✏️ Korrigiertes Bild gespeichert: {path}")

    # Neue Klasse hinzufügen
    if new_class:
        neue_klasse = st.text_input("🆕 Neue Klasse eingeben")
        if st.button("➕ Anlegen und Bild speichern") and neue_klasse:
            classes_df = pd.read_csv(CLASSES_CSV, sep=';')
            if neue_klasse not in classes_df['class_name'].values:
                new_id = classes_df['label_id'].max() + 1
                classes_df = pd.concat([
                    classes_df,
                    pd.DataFrame({"label_id": [new_id], "class_name": [neue_klasse]})
                ], ignore_index=True)
                classes_df.to_csv(CLASSES_CSV, sep=';', index=False)
                st.success(f"📄 Neue Klasse hinzugefügt: {neue_klasse}")
            regeln_df = pd.read_csv(RULES_CSV, sep=';')
            if neue_klasse not in regeln_df['klasse'].values:
                neue_regel = pd.DataFrame({
                    "klasse": [neue_klasse],
                    "min_um_hoehe": [0.0],
                    "max_um_hoehe": [9999.0]
                })
                regeln_df = pd.concat([regeln_df, neue_regel], ignore_index=True)
                regeln_df.to_csv(RULES_CSV, sep=';', index=False)
            ordner = os.path.join(NEUE_KLASSEN_DIR, neue_klasse)
            os.makedirs(ordner, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(ordner, f"{ts}.jpg")
            cv2.imwrite(path, frame)
            st.success(f"📸 Neue Klasse gespeichert: {path}")

cap.release()