import streamlit as st
import tempfile
from src.inference import predict_command

st.title("Speech Command Recognition Demo")

uploaded = st.file_uploader("Upload a WAV file (1 second)", type=["wav"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    label, probs = predict_command(tmp_path)

    st.success(f"Predicted command: {label}")