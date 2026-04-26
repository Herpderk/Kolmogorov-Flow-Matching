from huggingface_hub import hf_hub_download


def download_train_set(filename: str) -> str:
    return hf_hub_download(
        repo_id="ayz2/temporal_pdes",
        filename=filename,  # "kolmogorov/train.h5", # Adjust this path based on the exact file tree in the repo
        repo_type="dataset",
    )


def download_eval_set(filename: str) -> str:
    return hf_hub_download(
        repo_id="ayz2/temporal_pdes",
        filename=filename,  # "kolmogorov/test.h5",
        repo_type="dataset",
    )
