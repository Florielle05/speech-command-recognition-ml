import streamlit as st
import tempfile
import numpy as np
import soundfile as sf

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.inference import predict_command

st.title("Speech Command Recognition Demo")

st.write("Record from your microphone or upload a 1s WAV file.")

# --- Micro recording (WebRTC) ---
st.subheader("🎙️ Record from microphone (recommended)")

ctx = webrtc_streamer(
    key="speech-demo",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
)

tmp_path = None

if ctx.audio_receiver:
    if st.button("Use last recorded audio"):
        audio_frames = []
        # Drain buffered frames (short recording)
        while True:
            frame = ctx.audio_receiver.get_frame(timeout=0.1)
            if frame is None:
                break
            audio_frames.append(frame)

        if len(audio_frames) == 0:
            st.warning("No audio captured. Try recording again.")
        else:
            # Convert frames to mono float32 numpy
            samples = []
            sample_rate = None
            for f in audio_frames:
                arr = f.to_ndarray()
                # arr shape: (channels, samples)
                if arr.ndim == 2:
                    mono = arr.mean(axis=0)
                else:
                    mono = arr
                samples.append(mono.astype(np.float32))
                if sample_rate is None:
                    sample_rate = f.sample_rate

            audio_np = np.concatenate(samples)

            # Write WAV
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
