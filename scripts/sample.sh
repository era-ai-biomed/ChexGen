#!/bin/bash
#############################################################
# Image Generation Script - Generate CXR from trained diffusion model
#############################################################

# Set color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration parameters
NUM_GPUS=1
CONFIG_FILE="configs/model.py"
CKPT="weights/finetune_impression_512.pth"
OUTPUT_DIR="visualization"
PORT=12345
CFG_SCALE=4
SEED=1234

declare -a PROMPTS=(

    "Moderate cardiomegaly and findings suggesting mild vascular congestion."
    
    "Small bilateral pleural effusions. No focal consolidation identified."

    "Trace right pleural effusion. No parenchymal opacities to suggest pneumonia."

    "Lungs are clear. Heart is normal size probably exaggerated by mediastinal fat. No pleural abnormality. Solitary calcified granuloma left midlung unchanged. No evidence of intrathoracic malignancy."

    "Small well-defined right upper zone mass. Right main and lobar bronchi and pulmonary vessels can be seen (hilum overlay sign)."
    
)

# Display configuration information
echo -e "${GREEN}========== Generation Configuration ==========${NC}"
echo -e "${YELLOW}Number of GPUs:${NC} $NUM_GPUS"
echo -e "${YELLOW}Config file:${NC} $CONFIG_FILE" 
echo -e "${YELLOW}Checkpoint:${NC} $CKPT"
echo -e "${YELLOW}Output directory:${NC} $OUTPUT_DIR"
echo -e "${YELLOW}CFG scale:${NC} $CFG_SCALE"
echo -e "${YELLOW}Random seed:${NC} $SEED"
echo -e "${YELLOW}Number of prompts:${NC} ${#PROMPTS[@]}"
echo -e "${GREEN}===============================================${NC}"

# Display prompts
echo -e "${GREEN}========== Text Prompts Used for Generation ==========${NC}"
for i in "${!PROMPTS[@]}"; do
    echo -e "${BLUE}Prompt $((i)):${NC}"
    echo "${PROMPTS[$i]}"
    echo
done
echo -e "${GREEN}===============================================${NC}"

# Start generation
echo -e "${GREEN}Starting image generation...${NC}"

# Run the generation command
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${PORT} \
    tools/sample.py ${CONFIG_FILE} ${CKPT} \
    --work-dir ${OUTPUT_DIR} \
    --text-prompt "${PROMPTS[@]}" \
    --seed ${SEED} \
    --cfg-scale ${CFG_SCALE}

# Check generation results
echo -e "${GREEN}Generation complete! Images saved to: ${OUTPUT_DIR}${NC}"


