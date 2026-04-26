"""
# 2. Initialize Evaluation Dataset (Pass the TRAIN stats here!)
# We MUST use the training mean/std on the eval set to prevent data leakage
# and ensure the model sees identical scaling.
eval_dataset = NormalizedKolmogorovDataset(
    eval_path,
    k_frames=4,
    mean=train_dataset.mean,
    std=train_dataset.std
)
"""
