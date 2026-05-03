from huggingface_hub import hf_hub_download


def download_train_set(local_save_dir: str) -> str:
    return hf_hub_download(
        repo_id="ayz2/temporal_pdes",
        filename="train/KolmFlow_train_1024.h5",  # "kolmogorov/train.h5", # Adjust this path based on the exact file tree in the repo
        repo_type="dataset",
        local_dir=local_save_dir,
    )


def download_valid_set(local_save_dir: str) -> str:
    return hf_hub_download(
        repo_id="ayz2/temporal_pdes",
        filename="valid/KolmFlow_valid_256.h5",  # "kolmogorov/test.h5",
        repo_type="dataset",
        local_dir=local_save_dir,
    )
