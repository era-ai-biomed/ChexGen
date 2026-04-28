from radiffuser.datasets.t2i import T2IDataset, ImageDataset
from diffusers.models import AutoencoderKL

import os
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# Define the argument parser
parser = argparse.ArgumentParser(description='Image Embedding Generation')

parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for data loading')
parser.add_argument('--data_path', type=str, required=True, help='Path to the image data')
parser.add_argument('--data_csv', type=str, required=False, help='Path to the CSV file containing metadata')
parser.add_argument('--embedding_save_dir', type=str, required=True, help='Directory to save the image embeddings')
parser.add_argument('--parts', type=int, default=2, help='Number of parts to split the dataset')
parser.add_argument('--index', type=int, default=0, help='Index of the part to process')

args = parser.parse_args()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the embedding save directory if it doesn't exist
if not os.path.exists(args.embedding_save_dir):
    os.makedirs(args.embedding_save_dir, exist_ok=True)

# Define the transformations for image preprocessing
transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

# Create the dataset
dataset = ImageDataset(image_dir=args.data_path, transform=transform, return_name=True)

# Print the dataset size
print(f"Dataset size: {len(dataset)}")

# Calculate the indices for the part to process
part_size = len(dataset) // args.parts
start_index = args.index * part_size
end_index = start_index + part_size

# Filter the dataset based on the part indices
dataset.split_dataset(start_index, end_index)

# Calculate the total iterations
total_iterations = len(dataset) // args.batch_size
if len(dataset) % args.batch_size != 0:
    total_iterations += 1

# Create the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# Create the VAE model
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

# Initialize the progress bar and iterate over the data
pbar = tqdm(total=total_iterations)
for image, name in dataloader:
    image = image.to(device)

    with torch.no_grad():
        x = vae.encode(image).latent_dist.sample().mul_(0.18215)
        x = x.cpu().numpy()

    # Save the embedding for each image
    for i in range(len(name)):
        save_dict = {"image": x[i]}
        npz_save_dir = os.path.join(args.embedding_save_dir, name[i] + ".npz")
        dir_name = os.path.dirname(npz_save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        np.savez(npz_save_dir, **save_dict)

    pbar.update(1)
    pbar.set_postfix_str(f"Remaining: {len(dataset) - pbar.n}")

pbar.close()





