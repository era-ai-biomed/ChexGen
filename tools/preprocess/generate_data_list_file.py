"""Build the flat data-list .txt consumed by T2IDataset / T2IControlDataset.

Every emitted line is `<image>.npz <text>.npz [<cond>.npz]` — full paths
that the dataset loader opens directly.
"""
import argparse
import os

import mmengine


def main():
    parser = argparse.ArgumentParser(description="Build T2IDataset data list.")
    parser.add_argument("--dir-list", type=str, nargs="+", required=True,
                        help="One or more dataset directories that contain the embedding subfolders.")
    parser.add_argument("--target-folders", type=str, nargs="+",
                        default=["image_embedding_512", "caption_embedding"],
                        help="Embedding subfolders under each dir; add "
                             "'cond_embedding_512' for control models.")
    parser.add_argument("--save-file", type=str, required=True,
                        help="Output .txt path; parent dirs are created.")
    args = parser.parse_args()

    data_list = []
    for d in args.dir_list:
        embedd_cache_dirs = [os.path.join(d, folder) for folder in args.target_folders]
        image_embed_cache_dir = embedd_cache_dirs[0]
        image_names = mmengine.scandir(image_embed_cache_dir, suffix=".npz", recursive=True)

        sub_data_list = []
        for img_name in image_names:
            data = []
            exist = True
            for embedd_cache_dir in embedd_cache_dirs:
                embed_path = os.path.join(embedd_cache_dir, img_name)
                if not os.path.exists(embed_path):
                    exist = False
                    print(f"{embed_path} not exist!")
                data.append(embed_path)
            if exist:
                sub_data_list.append(data)
        data_list.extend(sub_data_list)
        print(f"Total matching entries in {d}: {len(sub_data_list)}")
    print(f"Total entries: {len(data_list)}")

    save_dir = os.path.dirname(args.save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(args.save_file, "w") as f:
        for item in data_list:
            f.write(" ".join(item) + "\n")


if __name__ == "__main__":
    main()
