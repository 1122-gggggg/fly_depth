# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:latest AS uv_build
# 使用 NVIDIA 官方 PyTorch 容器 (已內建 PyTorch + CUDA + cuDNN + TensorRT)
FROM nvcr.io/nvidia/pytorch:23.10-py3

# 複製 uv 執行檔
COPY --from=uv_build /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
# 指定 DNS，避免 build 時無法解析域名
ENV PIP_DEFAULT_TIMEOUT=100

# 1. 安裝系統基礎工具 (Olympe 編譯必備)
# apt 快取 mount: 下載過的 .deb 存在 host，重 build 不重下
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    libpcap-dev libavutil-dev libavcodec-dev libavformat-dev libswscale-dev \
    libraw1394-11 libavahi-client3 x11-apps \
    libgl1 \
    wget curl git

# 2. 複製預先下載的大型 wheel (onnxruntime-gpu 等)
COPY wheels/ /tmp/wheels/

# 3. 安裝 YOLO、OpenCV 與其他套件
# 先從本地 wheels/ 找，找不到再去 PyPI
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --find-links /tmp/wheels \
    --extra-index-url https://pypi.nchc.org.tw/simple \
    ultralytics opencv-python-headless numpy cupy-cuda12x onnxruntime-gpu

# 4. 單獨安裝 parrot-olympe (用 uv 加速下載)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --find-links /tmp/wheels \
    --extra-index-url https://pypi.nchc.org.tw/simple \
    parrot-olympe

# 4. 將 olympe_deps 的原生 .so 路徑加入動態連結器搜尋路徑
RUN python_ver=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    echo "/usr/local/lib/python${python_ver}/dist-packages/olympe_deps" \
    > /etc/ld.so.conf.d/olympe.conf && ldconfig

WORKDIR /workspace