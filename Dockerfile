FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

RUN pip install --no-cache-dir "numpy==1.26.4"

RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir "xformers==0.0.23.post1"

RUN git clone --branch v22.6.2 --depth 1 \
    https://github.com/bmaltais/kohya_ss.git /app/kohya
WORKDIR /app/kohya
RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir \
    "accelerate==0.25.0" \
    "diffusers[torch]==0.25.0" \
    "transformers==4.36.2" \
    "safetensors==0.4.2" \
    "huggingface-hub==0.20.1" \
    "opencv-python==4.7.0.68" \
    "lion-pytorch==0.0.6" \
    "lycoris_lora==2.0.2" \
    "open-clip-torch==2.20.0" \
    "pytorch-lightning==1.9.0" \
    "einops==0.7.0" \
    "ftfy==6.1.1" \
    "omegaconf==2.3.0" \
    "protobuf==3.20.3" \
    "rich==13.7.0" \
    "scipy==1.11.4" \
    "timm==0.6.12" \
    "toml==0.10.2" \
    "voluptuous==0.13.1" \
    "dadaptation==3.1" \
    "prodigyopt==1.0"

RUN pip install --no-cache-dir \
    "runpod==1.7.7" \
    "brotli" \
    "aiohttp" \
    "requests" \
    "bitsandbytes"

RUN mkdir -p /models/sdxl
RUN wget -q --show-progress \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    -O /models/sdxl/sd_xl_base_1.0.safetensors

WORKDIR /app
COPY handler.py /app/handler.py

RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
import xformers; print(f'xformers {xformers.__version__}'); \
import diffusers; print(f'diffusers {diffusers.__version__}'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK')"

CMD ["python", "/app/handler.py"]
