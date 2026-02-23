import streamlit as st
import tempfile
import numpy as np
import soundfile as sf

from src.inference import predict_command

TARGET_SR = 16000
TARGET_SEC = 1.0
TARGET_LEN = int(TARGET_SR * TARGET_SEC)

st.set_page_config(page_title="Speech Command Recognition Demo", layout="centered")
st.title("Speech Command Recognition Demo (Upload only)")
st.caption("Upload a short audio file. The app will convert to 1.0s mono @ 16kHz before inference.")


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32)
    # shape: (samples, channels)
    return x.mean(axis=1).astype(np.float32)


def resample_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(audio) == 0:
        return audio.astype(np.float32)
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    new_len = int(len(audio) * sr_out / sr_in)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def force_len(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n:
        return audio[:n]
    return np.pad(audio, (0, n - len(audio))).astype(np.float32)


def normalize(audio: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if m > 0:
        audio = audio / m
    return audio.astype(np.float32)


def preprocess_audio(path_in: str, path_out: str) -> dict:
    # Read with soundfile (WAV/FLAC/OGG supported; MP3 depends on your system)
    x, sr = sf.read(path_in, always_2d=False)
    x = np.asarray(x)

    # ensure float32
    if x.dtype.kind in ("i", "u"):
        # int -> float in [-1, 1] approx
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)

    # mono
    x = to_mono(x)

    # resample
    x = resample_linear(x, sr, TARGET_SR)
    sr2 = TARGET_SR

    # force 1 sec
    x = force_len(x, TARGET_LEN)

    # normalize
    x = normalize(x)

    sf.write(path_out, x, sr2)

    return {
        "sr_in": sr,
        "sr_out": sr2,
        "samples_out": len(x),
        "sec_out": len(x) / sr2,
    }


uploaded = st.file_uploader("Upload audio (WAV recommended)", type=["wav", "flac", "ogg", "mp3"])

if uploaded is None:
    st.info("Upload a file to run inference.")
    st.stop()

# Save uploaded file
with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp_in:
    tmp_in.write(uploaded.read())
    in_path = tmp_in.name

# Preprocess to a clean WAV (1s, 16kHz, mono)
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
    out_path = tmp_out.name

try:
    info = preprocess_audio(in_path, out_path)
except Exception as e:
    st.error("Failed to read/convert audio. Please try a WAV file.")
    st.exception(e)
    st.stop()

st.success("Audio converted ✅ (mono, 16kHz, 1.0s)")
st.write(info)

# Playback converted audio
audio_bytes = open(out_path, "rb").read()
st.audio(audio_bytes, format="audio/wav")

# Run inference
label, probs = predict_command(out_path)
st.success(f"Predicted command: {label}")

# Optional: show top-k probs if it's a dict or array
with st.expander("Show probabilities"):
    st.write(probs)
