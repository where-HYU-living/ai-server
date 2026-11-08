FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY templates ./templates
COPY scripts ./scripts
# 모델 파일은 AI 모드(INFERENCE=on)일 때만 이미지에 넣는다 (Hugging Face combe4259/chatatda-models)
ARG INFERENCE=off
RUN if [ "$INFERENCE" = "on" ]; then sh scripts/fetch_models.sh; fi
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
