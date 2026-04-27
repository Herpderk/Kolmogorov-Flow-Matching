import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from kolmogorov_flow_matching.dataset import NormalizedKolmogorovDataset
from kolmogorov_flow_matching.eval import visualize_frames
from kolmogorov_flow_matching.models.base import ConditionalGenerativeFramework


def train_conditional_generative_model(
    model: ConditionalGenerativeFramework,
    dataset: NormalizedKolmogorovDataset,
    optimizer: Optimizer,
    num_epochs: int,
    batch_size: int,
    max_grad_norm: float = 1.0,
    num_dataloader_workers: int = 4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
        pin_memory=True,
    )

    print("Starting training...")
    epoch_history = []
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # The 'with' statement ensures the bar safely closes before the print statement
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for batch in progress_bar:
                # Move data to GPU
                x_cond = batch["condition"].to(device)
                x_target = batch["target"].to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                loss = model.get_training_loss(x_target, x_cond)

                # Backward pass
                loss.backward()

                # Update weights
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()

                # Update progress bar statistics
                epoch_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}"})

        avg_loss = epoch_loss / len(train_loader)

        # 1. Store the metrics
        epoch_history.append(epoch + 1)
        loss_history.append(avg_loss)

        # 2. Clear the cell output (wait=True prevents flickering)
        clear_output(wait=True)

        # 3. Draw the updated plot
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_history, loss_history, marker="o", linestyle="-", color="b")
        plt.grid(linestyle="dashed")
        plt.title("Training Loss vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.show()

        # This will now print cleanly on a new line
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.5f}")

        # Evaluate model after each epoch
        model.eval()
        with torch.no_grad():
            rand_idx = torch.randint(0, len(dataset), (1,)).item()
            eval_sample = dataset[rand_idx]
            x_cond_eval = (
                torch.as_tensor(eval_sample["condition"]).unsqueeze(0).to(device)
            )
            x_target_eval = (
                torch.as_tensor(eval_sample["target"]).unsqueeze(0).to(device)
            )
            x_gen_eval = model.sample(x_cond_eval)
        target_frame = x_target_eval.detach().cpu().squeeze()
        gen_frame = x_gen_eval.detach().cpu().squeeze()
        comparison_tensor = torch.stack([target_frame, gen_frame], dim=0)
        print(f"Visualizing Random Sample #{rand_idx}")
        print(f"Target (Frame 1) vs. Generated Prediction (Frame 2):")
        visualize_frames(comparison_tensor)

    return avg_loss


def save_checkpoint(
    save_path: str,
    model: ConditionalGenerativeFramework,
    optimizer: Optimizer,
    final_epoch_loss: float,
    num_epochs: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": final_epoch_loss,
            "epoch": num_epochs,
        },
        save_path,
    )
    print(f"Checkpoint saved to {save_path}")
    print("-" * 50)
