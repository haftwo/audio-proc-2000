FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg alsa-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8020 \
    DATA_DIR=/data \
    HLS_DIR=/tmp/hls \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]
EXPOSE 8020
CMD ["python", "-m", "app.server"]
