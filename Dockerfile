FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget aria2 \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir xformers==0.0.23.post1 --no-deps

RUN pip install --no-cache-dir \
    accelerate==0.25.0 transformers==4.36.2 diffusers==0.25.0 \
    safetensors==0.4.1 opencv-python-headless==4.9.0.80 \
    einops==0.7.0 lion-pytorch==0.1.2 lycoris-lora==2.2.0.post3 \
    toml==0.10.2 voluptuous==0.14.1 huggingface-hub==0.20.3 \
    requests==2.31.0 runpod>=1.7.0 brotli aiohttp

RUN git clone https://github.com/kohya-ss/sd-scripts.git /kohya
WORKDIR /kohya
RUN git checkout sd3 || git checkout main
RUN pip install --no-cache-dir -e . --no-deps 2>/dev/null || true

RUN mkdir -p /models/sdxl
RUN aria2c --console-log-level=error -x 16 -s 16 \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    -d /models/sdxl -o sd_xl_base_1.0.safetensors

COPY handler.py /handler.py
CMD ["python", "/handler.py"]
