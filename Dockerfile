FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG EXTRAS
ARG HF_PRECACHE_DIR
ARG HF_TKN_FILE

ENV HF_HOME="/app/tmp/cache/huggingface"
ENV HF_HUB_CACHE="/app/tmp/cache/huggingface/hub"
ENV XDG_CACHE_HOME="/app/tmp/cache/huggingface"
ENV LIBROSA_CACHE_DIR="/app/tmp/librosa_cache"
ENV NUMBA_CACHE_DIR="/app/tmp/numba_cache"
ENV HF_HUB_ETAG_TIMEOUT="600"
ENV HF_HUB_DOWNLOAD_TIMEOUT="600"

# Install system dependencies
RUN apt-get update && \
   apt-get install -y ffmpeg git && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*

# 2) Install system dependencies + Python + pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        ffmpeg \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY . .

RUN mkdir -p $HF_HOME && \
    mkdir -p $HF_HUB_CACHE && \
    mkdir -p $XDG_CACHE_HOME && \
    mkdir -p $LIBROSA_CACHE_DIR && \
    mkdir -p $NUMBA_CACHE_DIR && \
    chmod 777 $HF_HOME && \
    chmod 777 $HF_HUB_CACHE && \
    chmod 777 $XDG_CACHE_HOME && \
    chmod 777 $LIBROSA_CACHE_DIR && \
    chmod 777 $NUMBA_CACHE_DIR && \
    chmod -R 777 /app

# Install WhisperLiveKit directly, allowing for optional dependencies
#   Note: For gates modedls, need to add your HF toke. See README.md
#         for more details.
RUN if [ -n "$EXTRAS" ]; then \
      echo "Installing with extras: [$EXTRAS]"; \
      pip install --no-cache-dir .[$EXTRAS]; \
    else \
      echo "Installing base package only"; \
      pip install --no-cache-dir .; \
    fi

# Enable in-container caching for Hugging Face models by: 
# Note: If running multiple containers, better to map a shared
# bucket. 
#
# A) Make the cache directory persistent via an anonymous volume.
#    Note: This only persists for a single, named container. This is 
#          only for convenience at de/test stage. 
#          For prod, it is better to use a named volume via host mount/k8s.
# VOLUME ["/root/.cache/huggingface/hub"]

# or
# B) Conditionally copy a local pre-cache from the build context to the 
#    container's cache via the HF_PRECACHE_DIR build-arg.
#    WARNING: This will copy ALL files in the pre-cache location.

# Conditionally copy a cache directory if provided
RUN if [ -n "$HF_PRECACHE_DIR" ]; then \
      echo "Copying Hugging Face cache from $HF_PRECACHE_DIR"; \
      mkdir -p /root/.cache/huggingface/hub && \
      cp -r $HF_PRECACHE_DIR/* /root/.cache/huggingface/hub; \
    else \
      echo "No local Hugging Face cache specified, skipping copy"; \
    fi

# Conditionally copy a Hugging Face token if provided

RUN if [ -n "$HF_TKN_FILE" ]; then \
      echo "Copying Hugging Face token from $HF_TKN_FILE"; \
      mkdir -p /root/.cache/huggingface && \
      cp $HF_TKN_FILE /root/.cache/huggingface/token; \
    else \
      echo "No Hugging Face token file specified, skipping token setup"; \
    fi
    
# Expose port for the transcription server
EXPOSE 8000

ENTRYPOINT ["whisperlivekit-server", "--host", "0.0.0.0"]

# Default args
CMD ["--model", "tiny", "--language", "en"]
#CMD ["--model", "projecte-aina/faster-whisper-large-v3-ca-3catparla", "--language", "ca"]