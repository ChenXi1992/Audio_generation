import torch
from torch.utils.data import DataLoader
from outdated.audio_transformer import AudioTransformer
from utils.data_loader import create_data_loader
from utils.visualization import plot_spectrogram

def evaluate(model, data_loader, criterion):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): Trained transformer model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): Loss function.
    
    Returns:
        avg_loss (float): Average loss over the dataset.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            # Prepare input and target
            src = batch[:, :-1, :]  # Input sequence (all but the last time step)
            tgt = batch[:, 1:, :]   # Target sequence (all but the first time step)

            # Forward pass
            output = model(src, tgt)
            loss = criterion(output, tgt)
            total_loss += loss.item()

            # Visualize the first batch
            if plot_spectrogram:
                plot_spectrogram(output[0].cpu().numpy(), title="Generated Spectrogram")

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Example usage
if __name__ == "__main__":
    # Load model and data
    model = AudioTransformer(input_dim=128, model_dim=256, num_heads=8, num_layers=6, max_length=1000)
    checkpoint = torch.load("outputs/checkpoints/model_checkpoint.pt")
    model.load_state_dict(checkpoint["state_dict"])

    # Create DataLoader
    data_dir = "data/processed"
    val_loader = create_data_loader(data_dir, batch_size=16, shuffle=False)

    # Evaluate
    criterion = torch.nn.MSELoss()
    avg_loss = evaluate(model, val_loader, criterion)
    print(f"Validation Loss: {avg_loss}")