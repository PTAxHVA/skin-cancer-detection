# HF Spaces (Docker SDK) chay Streamlit app. Cong 8501 (khop app_port o README).
FROM python:3.11-slim

# libgomp1: runtime OpenMP cho xgboost; curl cho healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl && rm -rf /var/lib/apt/lists/*

# HF Spaces yeu cau chay duoi user khong phai root
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Cai dependencies (he thong, user dung lai duoc)
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy toan bo source
COPY --chown=user . .

USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py"]
