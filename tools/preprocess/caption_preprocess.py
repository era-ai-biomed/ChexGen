import shutil
import io
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
try:
    from petrel_client.client import Client
except:
    pass
from radiffuser.models.t5 import T5Embedder


def split_data(data, parts):
    chunk_size = len(data) // parts
    remainder = len(data) % parts
    result = []
    start = 0
    for i in range(parts):
        end = start + chunk_size
        if i < remainder:
            end += 1
        result.append(data[start:end])
        start = end
    return result


if __name__ == '__main__':
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description='Caption Embedding Generation')
    parser.add_argument('--csv_dir', type=str,
        help='Path to the CSV file containing image captions')
    parser.add_argument('--embedding_save_dir', type=str,
        help='Directory to save the caption embeddings')
    parser.add_argument('--token-nums',
                        type=int,
                        default=180,
                        help='Number of tokens in the input sequence')
    parser.add_argument('--parts',
                        type=int,
                        default=1,
                        help='Number of subsets to split the data')
    parser.add_argument('--index',
                        type=int,
                        default=1,
                        help='Index of the subset to process')
    parser.add_argument('--n',
                        type=int,
                        default=400,
                        help='Number of captions to process at a time')
    parser.add_argument('--key',
                        type=str,
                        default='caption',
                        help='Key of the caption in the CSV file')
    args = parser.parse_args()

    try:
        file_client_args = {"backend": "petrel"}
        file_client = Client(**file_client_args)
    except:
        pass
    
    # Read CSV
    df = pd.read_csv(args.csv_dir)
    image_caption_list = list(zip(df['name'], df[args.key]))

    subsets = split_data(image_caption_list, args.parts)
    subset = subsets[args.index]

    # print the number of image_caption_list and subset
    print(f"CSV: {args.csv_dir}")
    print(f"Uizig {args.key}")
    print(f"Using: {len(image_caption_list)}")
    print(f"Subset: {len(subset)}")
    print(f"Token nums: {args.token_nums}")
    print((f"Processing subset {args.index} of {args.parts} subsets"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_model = T5Embedder(device=device)
    
    pbar = tqdm(total=len(subset), desc='Processing captions')

    
    os.makedirs(args.embedding_save_dir, exist_ok=True)

    mask_nums = []

    pbar = tqdm(total=len(subset), desc='Processing captions')
    for i in range(0, len(subset), args.n):
        batch = subset[i:i + args.n]
        captions = [
            caption for _, caption in batch if isinstance(caption, str)
        ]
        captions = [caption.replace("\n", "") for caption in captions]

        caption_embedding, caption_mask = text_model.get_text_embeddings(
            captions, token_nums=args.token_nums)

        mask_nums.extend(caption_mask.sum(dim=1).tolist())

        for j, (image_name, _) in enumerate(batch):
            save_path = os.path.join(args.embedding_save_dir, f"{image_name}.npz")

            # if not file_client.contains(save_path):
            #     with io.BytesIO() as f:
            #         np.savez(
            #             f,
            #             caption_embedding=caption_embedding[j].cpu().float().numpy(),
            #             caption_mask=caption_mask[j].cpu().float().numpy()
            #             )
            #         f.seek(0) 
            #         file_client.put(save_path, f)
            # else:
            #     data_bytes = file_client.get(save_path)
            #     try:
            #         with io.BytesIO(data_bytes) as data_buffer:
            #             data = np.load(data_buffer, allow_pickle=True)
            #             data = data['caption_embedding']
            #     except:
            #         with io.BytesIO() as f:
            #             np.savez(
            #                 f,
            #                 caption_embedding=caption_embedding[j].cpu().float().numpy(),
            #                 caption_mask=caption_mask[j].cpu().float().numpy()
            #                 )
            #             f.seek(0) 
            #             file_client.put(save_path, f)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                caption_embedding=caption_embedding[j].cpu().float().numpy(),
                caption_mask=caption_mask[j].cpu().float().numpy())

        pbar.update(args.n)
    pbar.close()

    # print the max, min, and average of the caption embedding mask, also 99.5%
    # covert list to array
    mask_nums = np.array(mask_nums)
    print(f"Caption embedding mask: len={len(mask_nums)}")
    print(f"Max: {mask_nums.max()}")
    print(f"Min: {mask_nums.min()}")
    print(f"Average: {mask_nums.mean()}")
    print(f"Median: {np.median(mask_nums)}")
    print(f"90%: {np.percentile(mask_nums, 90)}")