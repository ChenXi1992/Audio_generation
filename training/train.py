# training/train.py 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.decoder_transformer import DecoderOnlyAudioTransformer
from utils.data_loader import create_data_loader
from utils.visualization import plot_loss_curves
import os
import pickle

def train_model(config):
    """
    Train the decoder-only transformer model.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters and paths.
    """

    # Set device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f"Using device: {device}") 
    if torch.cuda.is_available(): 
        print(f"GPU: {torch.cuda.get_device_name(0)}") 

    # Extract hyperparameters
    input_dim = config["input_dim"] 
    model_dim = config["model_dim"] 
    num_heads = config["num_heads"] 
    num_layers = config["num_layers"] 
    max_length = config["max_length"] 
    batch_size = config["batch_size"] 
    learning_rate = config["learning_rate"] 
    num_epochs = config["num_epochs"] 
    data_dir = config["data_dir"] 
    checkpoint_dir = config["checkpoint_dir"] 
    early_stop_patience = config.get("early_stop_patience", 3)  # Stop training if no improvement for `patience` epochs

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model, loss function, and optimizer
    model = DecoderOnlyAudioTransformer(input_dim, model_dim, num_heads, num_layers, max_length)
    model = model.to(device) 
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader
    dataset = create_data_loader(data_dir, batch_size=batch_size, shuffle=True)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Batch size: {batch_size}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    # Lists to store losses for visualization
    train_losses = []
    val_losses = []

    # Keep track of the best checkpoint
    best_val_loss = float('inf')
    best_checkpoint_path = None
    epochs_no_improve = 0  # Track consecutive epochs without improvement

    # Training loop
    for epoch in range(num_epochs): 
        model.train() 
        total_train_loss = 0 

        for batch in train_loader:
            # Move batch to GPU
            batch = batch.to(device)

            # Prepare input and target (shifted by one time step)
            tgt = batch[:, :-1, :].to(device)  # Move to device
            target = batch[:, 1:, :].to(device)  # Move to device

            # Forward pass
            output = model(tgt)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Compute average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                tgt = batch[:, :-1, :].to(device)
                target = batch[:, 1:, :].to(device)
                output = model(tgt)
                loss = criterion(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset counter

            # Save the best checkpoint
            best_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_val_loss_{best_val_loss}_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_loss,
            }, best_checkpoint_path)
            print(f"Best checkpoint updated: {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Save losses
    with open('outputs/loss/train_losses.pkl', 'wb') as file:
        pickle.dump(train_losses, file)
    with open('outputs/loss/val_losses.pkl', 'wb') as file:
        pickle.dump(val_losses, file)

    # Plot the training and validation losses
    plot_loss_curves(train_losses, val_losses)
