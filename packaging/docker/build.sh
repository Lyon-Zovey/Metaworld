#!/usr/bin/env bash
# Build the Metaworld pipeline-runner Docker image (python + uv mode).
#
# Steps:
#   1. Snapshot the host `metaworld` conda env -> requirements.lock.txt + env.yaml
#      (committed, human-readable).
#   2. Filter the lock into a clean requirements.txt the docker build consumes
#      (drops conda-only packages, editable installs, local-path wheels).
#   3. docker build -f packaging/docker/Dockerfile (context = repo root, so the
#      `COPY . /opt/metaworld` line picks up the whole repo, minus .dockerignore).
#   4. (optional) docker save -> portable archive tarball.
#
# Usage:
#   bash packaging/docker/build.sh                  # build image
#   bash packaging/docker/build.sh --tag 0.2        # custom tag
#   bash packaging/docker/build.sh --save           # also produce metaworld-pipeline-<tag>.tar.gz
#   SKIP_FREEZE=1 bash packaging/docker/build.sh    # reuse existing requirements.txt

set -euo pipefail

# ---- args ----
TAG="0.1"
DO_SAVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)  TAG="$2"; shift 2 ;;
        --save) DO_SAVE=1; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- paths ----
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKER_DIR="${REPO}/packaging/docker"
ENV_NAME="${ENV_NAME:-metaworld}"
CONDA_BIN="${CONDA_BIN:-/mnt2/liangzhuowei/miniconda3/bin/conda}"
ENV_PYTHON="${CONDA_BIN%/conda}/../envs/${ENV_NAME}/bin/python"

cd "${REPO}"

# ---- 1. snapshot env (committed, for humans) ----
if [[ "${SKIP_FREEZE:-0}" == "1" && -f "${DOCKER_DIR}/requirements.txt" ]]; then
    echo "[1/3] SKIP_FREEZE=1 -> reusing ${DOCKER_DIR}/requirements.txt"
else
    echo "[1/3] freezing host env '${ENV_NAME}' -> requirements.lock.txt + env.yaml"
    "${ENV_PYTHON}" -m pip freeze > "${DOCKER_DIR}/requirements.lock.txt"
    "${CONDA_BIN}" env export -n "${ENV_NAME}" --no-builds > "${DOCKER_DIR}/env.yaml"

    # Filter the lock into the clean requirements.txt that docker build consumes:
    #   - drop editable installs (`-e ...` / `# Editable install ...`) — metaworld
    #     itself is installed inside the Dockerfile via `pip install -e /opt/metaworld`
    #   - drop local-path wheels (`name @ file:///...`)
    #   - drop conda's own bookkeeping packages
    echo "[1/3] filtering -> requirements.txt"
    grep -vE '^(-e |# Editable install)' "${DOCKER_DIR}/requirements.lock.txt" \
      | grep -vE '@ file://' \
      | grep -vEi '^(conda|conda-content-trust|conda-libmamba-solver|conda-package-handling|conda-package-streaming|libmambapy|mamba|menuinst|anaconda-)' \
      > "${DOCKER_DIR}/requirements.txt"
fi

# ---- 2. docker build ----
IMAGE="metaworld-pipeline:${TAG}"
echo "[2/3] docker build -t ${IMAGE}  (context=${REPO})"
docker build \
    -f "${DOCKER_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${REPO}"

# ---- 3. optional save ----
if [[ "${DO_SAVE}" == "1" ]]; then
    OUT="${REPO}/metaworld-pipeline-${TAG}.tar.gz"
    echo "[3/3] docker save -> ${OUT}"
    docker save "${IMAGE}" | gzip > "${OUT}"
    ls -lh "${OUT}"
else
    echo "[3/3] skip save (pass --save to produce a portable tarball)"
fi

echo
echo "Done."
echo "  image:  ${IMAGE}"
echo "  size:   $(docker images --format '{{.Size}}' "${IMAGE}")"
echo
echo "Next:"
echo "  bash packaging/docker/run.sh shell                    # interactive"
echo "  bash packaging/docker/run.sh pipeline ENV_NAME=hammer-v3 NUM_EPISODES=5"
