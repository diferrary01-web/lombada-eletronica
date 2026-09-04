# Imagem CPU. Para GPU NVIDIA, troque a base por
# `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, instale o python e use o
# extra `gpu` no lugar de `lpr`.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# PYTHONUNBUFFERED nao e detalhe: sem ele o stdout sai em blocos e o timestamp
# do `docker logs -t` passa a ser o do FLUSH, nao o do evento. Diagnosticar
# taxa de quadros ou intervalo entre passagens com log bufferizado leva a
# conclusao errada.

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[video,detect,lpr]"

COPY config/cameras.example.yaml ./config/

VOLUME ["/app/data", "/app/config"]

ENTRYPOINT ["lombada"]
CMD ["run", "--config", "config/cameras.yaml"]
