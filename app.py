import streamlit as st
import tempfile
import numpy as np
import soundfile as sf

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from src.inference import predict_command

TARGET_SR = 16000
TARGET_SEC = 1.0
TARGET_LEN = int(TARGET_SR * TARGET_SEC)

st.title("Speech Command Recognition Demo")

if "audio_chunks" not in st.session_state:
    st.session_state.audio_chunks = []
if "last_sr" not in st.session_state:
    st.session_state.last_sr = None

def audio_frame_callback(frame):
    arr = frame.to_ndarray()

    # arr can be (channels, samples) or (samples, channels)
    if arr.ndim == 2:
        # assume channel dimension is the smaller one (1 or 2)
        if arr.shape[0] <= 2:
            mono = arr.mean(axis=0)
        else:
            mono = arr.mean(axis=1)
    else:
        mono = arr

    mono = mono.astype(np.float32)

    st.session_state.audio_chunks.append(mono)
    st.session_state.last_sr = frame.sample_rate
    return frame

st.subheader("🎙️ Record from microphone")
ctx = webrtc_streamer(
    key="speech-demo",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
)

st.caption("1) Clique Start, parle, puis Stop. 2) Clique ‘Use recorded audio’.")

tmp_path = None

col1, col2 = st.columns(2)
with col1:
    if st.button("Use recorded audio"):
        if not st.session_state.audio_chunks or st.session_state.last_sr is None:
            st.warning("Aucun audio capturé. Clique Start, parle, puis Stop.")
        else:
            audio = np.concatenate(st.session_state.audio_chunks)
            sr = st.session_state.last_sr

            # --- optional: resample to TARGET_SR ---
            if sr != TARGET_SR:
                # lightweight resample without extra deps (linear)
                x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
                x_new = np.linspace(0, 1, num=int(len(audio) * TARGET_SR / sr), endpoint=False)
                audio = np.interp(x_new, x_old, audio).astype(np.float32)
                sr = TARGET_SR

            # --- force 1 second ---
            if len(audio) >= TARGET_LEN:
                audio = audio[:TARGET_LEN]
            else:
                audio = np.pad(audio, (0, TARGET_LEN - len(audio)))

            # simple normalize (avoid division by 0)
            m = np.max(np.abs(audio))
            if m > 0:
                audio = audio / m

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio, sr)
                tmp_path = tmp.name

with col2:
    if st.button("Clear buffer"):
        st.session_state.audio_chunks = []
        st.session_state.last_sr = None
        st.info("Buffer effacé.")

st.subheader("⬆️ Or upload a WAV")
uploaded = st.file_uploader("Upload a WAV file (1 second)", type=["wav"])
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

if tmp_path is not None:
    label, probs = predict_command(tmp_path)
    st.success(f"Predicted command: {label}")
