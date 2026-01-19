import streamlit as st
import tempfile
import numpy as np
import soundfile as sf
import queue

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.inference import predict_command

st.title("Speech Command Recognition Demo")
st.write("Record from your microphone or upload a 1s WAV file.")

# --- Micro recording ---
st.subheader("🎙️ Record from microphone")

ctx = webrtc_streamer(
    key="speech-demo",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
)

tmp_path = None

if ctx.audio_receiver:
    st.caption("1) Click Start (in the WebRTC widget), speak, then click Stop. 2) Click the button below.")

    if st.button("Use recorded audio"):
        audio_frames = []
        sample_rate = None

        # Drain whatever is currently buffered
        while True:
            try:
                f = ctx.audio_receiver.get_frame(timeout=0.2)
            except queue.Empty:
                break  # no more frames available right now

            audio_frames.append(f)
            if sample_rate is None:
                sample_rate = f.sample_rate

        if not audio_frames:
            st.warning("No audio captured. Make sure you clicked Start, spoke, then Stop.")
        else:
            samples = []
            for f in audio_frames:
                arr = f.to_ndarray()
                # arr is typically (channels, samples)
                if arr.ndim == 2:
                    mono = arr.mean(axis=0)
                else:
                    mono = arr
                samples.append(mono.astype(np.float32))

            audio_np = np.concatenate(samples)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_np, sample_rate)
                tmp_path = tmp.name

# --- Upload fallback ---
st.subheader("⬆️ Or upload a WAV")
uploaded = st.file_uploader("Upload a WAV file (1 second)", type=["wav"])
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

# --- Inference ---
if tmp_path is not None:
    label, probs = predict_command(tmp_path)
    st.success(f"Predicted command: {label}")
