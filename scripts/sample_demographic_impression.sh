#!/bin/bash
set -euo pipefail
#############################################################
# Sample chest X-rays conditioned on impression + sex/age/race demographics.
#
# Usage:
#   bash scripts/sample_demographic_impression.sh                       # default: data/mimic_val_p19_demographic_impression_example.csv
#   bash scripts/sample_demographic_impression.sh path/to/prompts.csv   # custom CSV
#############################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration parameters
NUM_GPUS=1
TORCHRUN="/scratch/m000071/yfj/miniconda3/envs/chexgen/bin/torchrun"
CONFIG_FILE="configs/finetuned_demographic_impression_512.py"
CKPT="weights/finetuned_demographic_impression_512.pth"
PROMPT_FILE="${1:-data/mimic_val_p19_demographic_impression_example.csv}"
OUTPUT_DIR="visualization"
PORT=12348
CFG_SCALE=4
SEED=1234

# Display configuration information
echo -e "${GREEN}========== Generation Configuration ==========${NC}"
echo -e "${YELLOW}Number of GPUs:${NC} $NUM_GPUS"
echo -e "${YELLOW}Config file:${NC} $CONFIG_FILE"
echo -e "${YELLOW}Checkpoint:${NC} $CKPT"
echo -e "${YELLOW}Prompt file:${NC} $PROMPT_FILE"
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

echo -e "${GREEN}Starting image generation...${NC}"

"${TORCHRUN}" \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${PORT} \
    tools/sample.py "${CONFIG_FILE}" "${CKPT}" \
    --work-dir "${OUTPUT_DIR}" \
    --text-prompt-file "${PROMPT_FILE}" \
    --seed "${SEED}" \
    --cfg-scale "${CFG_SCALE}"

echo -e "${GREEN}Generation complete! Images saved to: ${OUTPUT_DIR}${NC}"
