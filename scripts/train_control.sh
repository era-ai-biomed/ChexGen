#!/bin/bash
set -euo pipefail
#############################################################
# Train ControlT2IDiT (mask + text -> image).
#
# Usage:
#   bash scripts/train_control.sh path/to/config.py
#   NUM_GPUS=4 bash scripts/train_control.sh path/to/config.py
#   bash scripts/train_control.sh path/to/config.py --pretrained weights/finetuned_impression_512.pth
#############################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [[ $# -lt 1 ]]; then
    echo -e "${RED}Usage:${NC} bash scripts/train_control.sh <config.py> [extra args]"
    exit 1
fi

CONFIG_FILE="$1"
shift

NUM_GPUS="${NUM_GPUS:-1}"
ACCELERATE=accelerate
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
WORK_DIR="${WORK_DIR:-work_dirs/$(basename "${CONFIG_FILE}" .py)}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo -e "${RED}Config not found:${NC} ${CONFIG_FILE}"
    exit 1
fi

echo -e "${GREEN}========== Training Configuration ==========${NC}"
echo -e "${YELLOW}Number of GPUs:${NC} $NUM_GPUS"
echo -e "${YELLOW}Mixed precision:${NC} $MIXED_PRECISION"
echo -e "${YELLOW}Config file:${NC} $CONFIG_FILE"
echo -e "${YELLOW}Work dir:${NC} $WORK_DIR"
echo -e "${GREEN}=============================================${NC}"

if [[ ${NUM_GPUS} -gt 1 ]]; then
    "${ACCELERATE}" launch --multi_gpu --num_processes "${NUM_GPUS}" --mixed_precision "${MIXED_PRECISION}" \
        tools/train_control.py "${CONFIG_FILE}" --work-dir "${WORK_DIR}" "$@"
else
    "${ACCELERATE}" launch --mixed_precision "${MIXED_PRECISION}" \
        tools/train_control.py "${CONFIG_FILE}" --work-dir "${WORK_DIR}" "$@"
fi

echo -e "${GREEN}Training finished.${NC}"
