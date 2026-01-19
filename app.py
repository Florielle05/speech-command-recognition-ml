import streamlit as st
import tempfile
from src.inference import predict_command
from audiorecorder import audiorecorder

st.title("Speech Command Recognition Demo")

st.subheader("1) Record from microphone")
audio = audiorecorder("🎙️ Record", "⏹️ Stop")

tmp_path = None

if len(audio) > 0:
    # streamlit-audiorecorder returns an AudioSegment (pydub)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        tmp_path = tmp.name

st.subheader("2) Or upload a WAV file")
uploaded = st.file_uploader("Upload a WAV file (1 second)", type=["wav"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

if tmp_path is not None:
    label, probs = predict_command(tmp_path)
    st.success(f"Predicted command: {label}")
