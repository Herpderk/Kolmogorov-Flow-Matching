"""
# 1. Initialize Training Dataset (computes stats automatically)
train_dataset = NormalizedKolmogorovDataset(train_path, k_frames=4)

# Set num_workers > 0 to read from disk asynchronously while the GPU computes
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True # Speeds up CPU-to-GPU transfers
)

# Usage matches the training loop we wrote earlier
for batch in train_loader:
    x_cond = batch['condition'].cuda()
    x_target = batch['target'].cuda()

    # ... pass to model.get_training_loss(x_target, x_cond)
"""

"""
# 1. Compute stats using the utility we wrote earlier
global_mean, global_std = compute_hdf5_stats(train_path, dataset_key='u')

# 2. Initialize the datasets, passing the explicit stats
train_dataset = NormalizedKolmogorovDataset(
    train_path, mean=global_mean, std=global_std
)
eval_dataset = NormalizedKolmogorovDataset(
    eval_path, mean=global_mean, std=global_std
)

# 3. Initialize the Model, baking the stats into the state_dict
unet = AgnosticUNet(data_shape=[1, 64, 64]).cuda()
model = ConditionalFlowMatching(
    unet,
    mean=global_mean,
    std=global_std
).cuda()

# (Later, during inference...)
# Condition is pulled from dataset (it is already normalized)
x_cond = eval_dataset[0]['condition'].unsqueeze(0).cuda()

# The model generates the prediction in latent space, but seamlessly
# outputs it in physical flow metrics!
physical_next_frame = model.sample(x_cond, method='dopri5', return_physical=True)

print("Output shape:", physical_next_frame.shape)
print("True physical scale recovered. Mean:", physical_next_frame.mean().item())
"""
