# Base image of the official evaluation: CUDA PyTorch and the starter-kit dependencies.
# Build it once with `docker build -t isu-challenge-base docker/`; your submission's
# Dockerfile starts with `FROM isu-challenge-base`.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Models downloaded while building an image go to HF_HOME, which stays readable when the
# evaluation runs the container as a non-root user without network access. The code of
# remote-code models is cached in /tmp, the only writable folder during the evaluation.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch-cache \
    HF_HOME=/opt/huggingface \
    HF_MODULES_CACHE=/tmp/hf-modules

COPY requirements.txt prefetch_weights.py /opt/isu/

# Keep the base image's torch build so the CUDA libraries stay consistent.
RUN python -c "import torch; print('torch==' + torch.__version__)" > /opt/isu/constraints.txt \
 && python -m pip install -c /opt/isu/constraints.txt -r /opt/isu/requirements.txt \
 && python /opt/isu/prefetch_weights.py \
 && mkdir -p "$HF_HOME" \
 && chmod -R a+rwX "$HF_HOME" && chmod -R a+rX /opt/torch-cache /opt/isu
