# Metaworld Pipeline Runner Image
#
# Strategy: classic pip + uv. Build context is the repo root; build.sh sets that.
#
# Build:
#   bash packaging/docker/build.sh
#   bash packaging/docker/build.sh --tag 0.2 --save
#
# Run:
#   bash packaging/docker/run.sh shell
#   bash packaging/docker/run.sh pipeline ENV_NAME=hammer-v3 NUM_EPISODES=5

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libs:
#   - EGL / OpenGL / OSMesa  -> mujoco rendering (MUJOCO_GL=egl|osmesa)
#   - libglib + libsm        -> opencv runtime
#   - ffmpeg                 -> rgb / sceneflow video codecs
#   - git                    -> some pip packages need it at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
        libglu1-mesa libgl1 libglx-mesa0 libosmesa6 libegl1 libgles2 \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv = Rust-rewritten pip, ~10-50x faster.
RUN pip install --no-cache-dir uv

# Install Python deps first so Docker layer cache is reused as long as
# requirements.txt doesn't change.
COPY packaging/docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

# Copy the metaworld repo into the image and install it as an editable package.
# Anything matching .dockerignore is skipped (datasets, .git, caches, ...).
COPY . /opt/metaworld
RUN uv pip install --system --no-cache -e /opt/metaworld

# Default mujoco rendering backend; switch to osmesa at runtime if no GPU.
ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONPATH=/opt/metaworld

WORKDIR /opt/metaworld

CMD ["bash"]
