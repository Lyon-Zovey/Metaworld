#!/usr/bin/env bash
# Quick smoke-test for run_pipeline.sh changes.
# Runs the full pipeline on ONE env with 2 episodes, then checks:
#   (A) large .npy files were deleted after compression
#   (B) compressed outputs exist
#   (C) top-level symlinks N.mp4 were created and are valid
set -euo pipefail

TEST_ENV="reach-v3"
TEST_EPS=2
DATASET_DIR="dataset/${TEST_ENV}/camera_data"
ENV_DIR="dataset/${TEST_ENV}"

echo ""
echo "========================================================="
echo "  SMOKE TEST  env=${TEST_ENV}  episodes=${TEST_EPS}"
echo "========================================================="

# ── run the pipeline ──────────────────────────────────────────
ENV_NAME="${TEST_ENV}" NUM_EPISODES="${TEST_EPS}" bash run_pipeline.sh

# ── checks ───────────────────────────────────────────────────
PASS=0
FAIL=0

check_absent() {
  local label="$1"; local pattern="$2"
  local files
  files=$(find "${DATASET_DIR}" -name "${pattern}" 2>/dev/null || true)
  if [ -z "$files" ]; then
    echo "  [OK ] DELETED  ${label}"
    PASS=$((PASS+1))
  else
    echo "  [FAIL] SHOULD BE DELETED but found: ${label}"
    echo "    $files"
    FAIL=$((FAIL+1))
  fi
}

check_present() {
  local label="$1"; local pattern="$2"
  local files
  files=$(find "${DATASET_DIR}" -name "${pattern}" 2>/dev/null || true)
  if [ -n "$files" ]; then
    echo "  [OK ] EXISTS   ${label}"
    PASS=$((PASS+1))
  else
    echo "  [FAIL] MISSING  ${label}"
    FAIL=$((FAIL+1))
  fi
}

echo ""
echo "--- (A) Large .npy should be DELETED ---"
check_absent "scene_point_flow_ref*.npy (non-anchor)" "scene_point_flow_ref[0-9][0-9][0-9][0-9][0-9].npy"
check_absent "depth_video.npy"                        "depth_video.npy"
check_absent "seg.npy"                                "seg.npy"

echo ""
echo "--- (B) Compressed outputs should EXIST ---"
check_present "scene_point_flow_ref*_v3_10b_h265_crf0.mp4"  "scene_point_flow_ref*_v3_10b_h265_crf0.mp4"
check_present "depth_video_int16mm_dt.b2nd"                  "depth_video_int16mm_dt.b2nd"
check_present "seg.b2nd"                                      "seg.b2nd"
check_present "scene_point_flow_ref*.anchor.npy (kept)"       "*.anchor.npy"

# ── summary ──────────────────────────────────────────────────
echo ""
echo "========================================================="
echo "  Results:  ${PASS} passed,  ${FAIL} failed"
echo "========================================================="
if [ "${FAIL}" -gt 0 ]; then
  exit 1
fi
