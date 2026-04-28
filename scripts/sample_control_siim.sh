#!/bin/bash
set -euo pipefail
#############################################################
# Sample chest X-rays conditioned on impression + SIIM pneumothorax mask
# (ControlNet-style spatial conditioning).
#
# Usage:
#   bash scripts/sample_control_siim.sh                       # default: data/siim_control_example.csv
#   bash scripts/sample_control_siim.sh path/to/prompts.csv   # custom CSV
#
# CSV columns required:
#   - name        (output filename stem)
#   - impression  (text prompt; override via TEXT_KEY env var)
#   - mask        (path to mask image; override via COND_KEY env var)
# Optional env: COND_DIR (base dir for relative mask paths).
#############################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration parameters
NUM_GPUS=1
TORCHRUN="/scratch/m000071/yfj/miniconda3/envs/chexgen/bin/torchrun"
CONFIG_FILE="configs/finetuned_control_siim_512.py"
CKPT="weights/finetuned_control_siim_512.pth"
PROMPT_FILE="${1:-data/siim_control_example.csv}"
TEXT_KEY="${TEXT_KEY:-impression}"
COND_KEY="${COND_KEY:-mask}"
COND_DIR="${COND_DIR:-data/siim_masks}"
OUTPUT_DIR="visualization"
PORT=12349
CFG_SCALE=4
SEED=1234

# Display configuration information
echo -e "${GREEN}========== Generation Configuration ==========${NC}"
echo -e "${YELLOW}Number of GPUs:${NC} $NUM_GPUS"
echo -e "${YELLOW}Config file:${NC} $CONFIG_FILE"
echo -e "${YELLOW}Checkpoint:${NC} $CKPT"
echo -e "${YELLOW}Prompt file:${NC} $PROMPT_FILE"
echo -e "${YELLOW}Text key:${NC} $TEXT_KEY"
echo -e "${YELLOW}Cond key:${NC} $COND_KEY"
echo -e "${YELLOW}Cond dir:${NC} ${COND_DIR:-<absolute paths>}"
echo -e "${YELLOW}Output directory:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}CFG scale:${NC} $CFG_SCALE"
echo -e "${YELLOW}Random seed:${NC} $SEED"
echo -e "${GREEN}===============================================${NC}"

if [[ ! -f "${CKPT}" ]]; then
    echo -e "${RED}Checkpoint not found:${NC} ${CKPT}"
    echo "Place the downloaded checkpoint at ${CKPT}, or edit CKPT in this script."
    exit 1
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo -e "${RED}Prompt file not found:${NC} ${PROMPT_FILE}"
    exit 1
fi

EXTRA_ARGS=()
if [[ -n "${COND_DIR}" ]]; then
    EXTRA_ARGS+=(--cond-dir "${COND_DIR}")
fi

echo -e "${GREEN}Starting image generation...${NC}"

"${TORCHRUN}" \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${PORT} \
    tools/sample_control.py "${CONFIG_FILE}" "${CKPT}" \
    --work-dir "${OUTPUT_DIR}" \
    --text-prompt-file "${PROMPT_FILE}" \
    --text-prompt-key "${TEXT_KEY}" \
    --cond-key "${COND_KEY}" \
    --seed "${SEED}" \
    --cfg-scale "${CFG_SCALE}" \
    "${EXTRA_ARGS[@]}"

echo -e "${GREEN}Generation complete! Images saved to: ${OUTPUT_DIR}${NC}"
