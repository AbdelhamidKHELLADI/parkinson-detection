FROM python:3.11-slim AS base

WORKDIR /app
COPY docker_requirements.txt .
RUN  pip install --no-cache-dir -r docker_requirements.txt
COPY predict/svm ./predict/svm
COPY utils/ ./utils
COPY app.py .
# Environment variables (example, change as needed)
ENV MODEL_PATH=/app/models/svm_fold3.pkl \
    TMP_PATH=/app/tmp/audio \
    TMP_FEATURES_PATH=/app/tmp/test.csv \
    ACOUSTIC_FEATURES_DIR_PATH=/app/features \
    UPLOAD_FOLDER=/app/uploads

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]