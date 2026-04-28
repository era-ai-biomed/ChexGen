import mmengine
import os

if __name__ == '__main__':

    base_dir = "/mnt/petrelfs/jiyuanfeng.p/data/"

    dir_list = [
        # '/mnt/proj76/ylchen/datasets/cxr14',
        # '/mnt/proj76/ylchen/datasets/chexpert',
        # '/mnt/proj76/ylchen/datasets/padchest',
        # '/mnt/proj76/ylchen/datasets/ranzcr',
        # '/mnt/proj76/ylchen/datasets/covid',
        # '/mnt/proj76/ylchen/datasets/vindr_pcxr',
        # '/mnt/proj76/ylchen/datasets/brax',
        # '/mnt/proj76/ylchen/datasets/brax',
        # '/mnt/sdb/yuanfengji/data/xraygen/downsteam/siim-acr-pneumothorax',
        # "/mnt/petrelfs/jiyuanfeng.p/data/mimic-cxr"
        # "/mnt/petrelfs/jiyuanfeng.p/data/vindr-ribcxr"
        # "/mnt/petrelfs/jiyuanfeng.p/data/vindr-cxr"
        "/mnt/petrelfs/jiyuanfeng.p/data/object-cxr"
        # "/mnt/petrelfs/jiyuanfeng.p/data/chex-det10"
    ]

    target_folders = ["image_embedding_1024", "caption_embedding", "condition_embedding_1024"]

    data_list = []
    for dir in dir_list:
        sub_data_list = []

        embedd_cache_dirs = [os.path.join(dir, folder) for folder in target_folders]

        image_embed_cache_dir = embedd_cache_dirs[0]

        image_names = mmengine.scandir(embedd_cache_dirs[0], suffix='.npz', recursive=True)
        for img_name in image_names:
            img_path = os.path.join(image_embed_cache_dir, img_name)
            data = []
            exist = True
            for embedd_cache_dir in embedd_cache_dirs:
                embed_path = os.path.join(embedd_cache_dir, img_name)
                if not os.path.exists(embed_path):
                    exist = False
                    print(f"{embed_path} not exist!")
                data.append(embed_path.replace(base_dir, ""))
            if exist:
                sub_data_list.append(data)
        data_list.extend(sub_data_list)
        print(f"Total number of matching data in {dir}: {len(sub_data_list)}")
    print("Total number of data: ", len(data_list))

    save_file = "/mnt/petrelfs/jiyuanfeng.p/data/meta/second_stage_object_cxr.txt"
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    # save to txt file, override
    with open(save_file, 'w') as f:
        for item in data_list:
            # get image and text path, separated by space
            item = ' '.join(item)
            f.write("%s\n" % item)
