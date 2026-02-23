import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
import io # Ajouté pour gérer les flux mémoire
from streamlit_mic_recorder import mic_recorder # À ajouter dans requirements.txt

from src.inference import predict_command

TARGET_SR = 16000
TARGET_SEC = 1.0
TARGET_LEN = int(TARGET_SR * TARGET_SEC)

st.set_page_config(page_title="Speech Command Recognition Demo", layout="centered")
st.title("Speech Command Recognition (Live & Upload)")
st.caption("Record or upload a short command. Auto-converted to 1.0s mono @ 16kHz.")

# --- Fonctions de traitement (Inchangées) ---
def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1: return x.astype(np.float32)
    return x.mean(axis=1).astype(np.float32)

def resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(audio) == 0: return audio.astype(np.float32)
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    new_len = int(len(audio) * sr_out / sr_in)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)

def force_len(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n: return audio[:n]
    return np.pad(audio, (0, n - len(audio))).astype(np.float32)

def normalize(audio: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if m > 0: audio = audio / m
    return audio.astype(np.float32)

def preprocess_audio(path_in: str, path_out: str) -> dict:
    x, sr = sf.read(path_in, always_2d=False)
    x = np.asarray(x)
    if x.dtype.kind in ("i", "u"):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)
    x = to_mono(x)
    x = resample_linear(x, sr, TARGET_SR)
    x = force_len(x, TARGET_LEN)
    x = normalize(x)
    sf.write(path_out, x, TARGET_SR)
    return {"sr_in": sr, "sr_out": TARGET_SR, "samples_out": len(x), "sec_out": len(x) / TARGET_SR}

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.write("Option 1 : Enregistrement")
    recorded_audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹️ Stop", key='recorder')

with col2:
    st.write("Option 2 : Téléchargement")
    uploaded = st.file_uploader("WAV, FLAC, MP3", type=["wav", "flac", "ogg", "mp3"])

# Source de données finale
audio_source = None
if recorded_audio:
    audio_source = recorded_audio['bytes']
elif uploaded:
    audio_source = uploaded.read()

if audio_source is None:
    st.info("Utilisez le micro ou téléchargez un fichier pour commencer.")
    st.stop()

# Gestion des fichiers temporaires
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
    tmp_in.write(audio_source)
    in_path = tmp_in.name

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
    out_path = tmp_out.name

try:
    info = preprocess_audio(in_path, out_path)
    st.success("Audio traité ✅")
    
    # Lecture & Prédiction
    st.audio(open(out_path, "rb").read(), format="audio/wav")
    label, probs = predict_command(out_path)
    
    st.metric("Commande prédite", label.upper())
    
    with st.expander("Probabilités détaillées"):
        st.write(probs)
        
except Exception as e:
    st.error("Erreur de traitement.")
    st.exception(e)
