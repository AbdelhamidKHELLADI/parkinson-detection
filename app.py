import streamlit as st
import os
from predict.svm.predict import run_prediction

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Parkinson Detection", page_icon="🧠", layout="centered")

st.title("🧠 Parkinson Detection from Voice")
st.write("Upload an audio file containing vowel sounds (a, e, i, o, u) for analysis.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
st.info("⚠️ Disclaimer: This tool is a research prototype. "
        "It is **not validated for medical use** and should not be considered a clinical diagnosis. "
        "For proper evaluation, please consult a healthcare professional.")

if uploaded_file is not None:

    filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(filepath, format="audio/wav")

    with st.spinner("Analyzing audio..."):
        result = run_prediction(filepath)

    if result == "healthy":
        st.success(f"✅ Result: {result}")
    else:
        st.warning(f"⚠️ Result: {result}")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #666;

    }
    </style>
    <div class="footer">
        Developed by <a href="https://www.linkedin.com/in/khelladi-abdelhamid/" target="_blank">Abdelhamid Khelladi</a> |
        Available on the project repo: <a href="https://github.com/AbdelhamidKHELLADI/parkinson-detection" target="_blank">parkinson-detection</a>
    </div>
    """,
    unsafe_allow_html=True
)