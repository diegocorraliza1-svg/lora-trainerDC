# ============================================================
# LoRA Trainer Worker — Dockerfile auditado
# Base: PyTorch 2.1.2 + CUDA 12.1 (compatible con Kohya v22.6.2)
# ============================================================
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---- Dependencias del sistema ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 1. Pinar numpy ANTES de todo (evita numpy 2.x) ----
RUN pip install --no-cache-dir "numpy==1.26.4"

# ---- 2. xformers compatible con torch 2.1.2 + CUDA 12.1 ----
RUN pip install --no-cache-dir "xformers==0.0.23.post1"

# ---- 3. Kohya sd-scripts v22.6.2 (desde source) ----
RUN git clone --branch v22.6.2 --depth 1 https://github.com/kohya-ss/sd-scripts.git /app/kohya
WORKDIR /app/kohya
RUN pip install --no-cache-dir -e .

# ---- 4. Dependencias pinadas de Kohya v22.6.2 ----
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
    "wandb==0.15.11" \
    "dadaptation==3.1" \
    "prodigyopt==1.0" \
    "invisible-watermark==0.2.0"

# ---- 5. RunPod SDK + dependencias del handler ----
RUN pip install --no-cache-dir \
    "runpod==1.7.7" \
    "brotli" \
    "aiohttp" \
    "supabase"

WORKDIR /app

# ---- 6. Copiar handler ----
COPY handler.py /app/handler.py

# ---- 7. Verificación de imports al build ----
RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}'); \
import xformers; print(f'xformers {xformers.__version__}'); \
import diffusers; print(f'diffusers {diffusers.__version__}'); \
import transformers; print(f'transformers {transformers.__version__}'); \
import accelerate; print(f'accelerate {accelerate.__version__}'); \
import cv2; print(f'opencv {cv2.__version__}'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK') \
"

CMD ["python", "/app/handler.py"]
