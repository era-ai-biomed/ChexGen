# Chexgen

A text-to-image diffusion model for chest X-ray (CXR) generation from textual descriptions.

## Environment Setup

### Requirements

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Radiffuser.git
cd Radiffuser
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# OR
venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Main dependencies include:
- torch>=2.2.0
- torchvision>=0.17.0
- diffusers>=0.14.0
- transformers==4.38.1
- xformers
- mmengine>=0.8.0

## Model Weights

Download the pre-trained model weights from Google Drive:
[Download model weights](https://drive.google.com/file/d/1pbYSzMYFkUps6-iuIizgxYDZ46lJMxSz/view?usp=sharing)

After downloading, place the weights file in your preferred location. The default path used in the sample script is:
```
weights/impression_512.pth
```

You may need to modify the path in the sample script to match your setup.

## Image Generation with sample.sh

The `sample.sh` script in the `scripts` directory provides a convenient way to generate chest X-ray images from textual descriptions.

### Script Overview

This bash script:
1. Configures generation parameters (GPU count, checkpoint path, output directory, etc.)
2. Takes predefined text prompts describing medical conditions
3. Runs the generation model via distributed PyTorch
4. Saves the generated images to the specified output directory

### Usage

1. Make the script executable:
```bash
chmod +x scripts/sample.sh
```

2. Update the script parameters if needed:
   - `NUM_GPUS`: Number of GPUs to use
   - `CONFIG_FILE`: Path to model configuration
   - `CKPT`: Path to model checkpoint
   - `OUTPUT_DIR`: Directory to save generated images
   - `CFG_SCALE`: Classifier-free guidance scale (controls adherence to text)
   - `SEED`: Random seed for reproducibility
   - `PROMPTS`: Array of text descriptions to generate images from

3. Run the script:
```bash
./scripts/sample.sh
```

### Default Text Prompts

The script includes several example prompts:
- "Moderate cardiomegaly and findings suggesting mild vascular congestion."
- "Small bilateral pleural effusions. No focal consolidation identified."
- "Trace right pleural effusion. No parenchymal opacities to suggest pneumonia."
- "Lungs are clear. Heart is normal size probably exaggerated by mediastinal fat. No pleural abnormality. Solitary calcified granuloma left midlung unchanged. No evidence of intrathoracic malignancy."
- "Small well-defined right upper zone mass. Right main and lobar bronchi and pulmonary vessels can be seen (hilum overlay sign)."

### Custom Text Prompts

To use your own prompts, modify the `PROMPTS` array in the script:

```bash
declare -a PROMPTS=(
    "Your custom prompt 1"
    "Your custom prompt 2"
    # ...
)
```

## Advanced Usage

For more advanced control, you can directly use the Python script:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    tools/sample.py configs/model.py PATH_TO_CHECKPOINT \
    --work-dir output_directory \
    --text-prompt "Your prompt here" \
    --seed 1234 \
    --cfg-scale 4.0
```

### Parameters:
- `--text-prompt`: Text descriptions for generation (multiple prompts can be provided)
- `--text-prompt-file`: Alternative to inline prompts, specify a file containing prompts
- `--work-dir`: Output directory for generated images 
- `--cfg-scale`: Guidance scale (higher values follow text more closely)
- `--num-sampling-steps`: Number of diffusion steps (default: 100)
- `--seed`: Random seed for reproducibility
- `--batch-size`: Batch size for generation

## Output

Generated images will be saved in the specified output directory (default: `visualization/`). 

