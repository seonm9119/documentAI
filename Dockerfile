FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    DOCUMENT_AI_DATA_DIR=/app/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py convert_to_img.py normalize.py ./
COPY api ./api

RUN mkdir -p /app/data/uploads /app/data/images /app/data/overlays

EXPOSE 8030

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8030"]
