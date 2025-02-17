import torch
import numpy as np
import librosa
import os
from outdated.audio_transformer import AudioTransformer  # Import your model class

def load_model(checkpoint_path, input_dim, model_dim, num_heads, num_layers, max_length):
    """
    Load the trained model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        input_dim (int): Input dimension of the model.
        model_dim (int): Model dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_length (int): Maximum sequence length.
    
    Returns:
        model (AudioTransformer): Loaded model.
    """
    # Initialize the model
    model = AudioTransformer(input_dim, model_dim, num_heads, num_layers, max_length)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    
    return model

def generate_spectrogram(model, seed_spectrogram, max_length):
    """
    Generate a spectrogram using the trained model.
    
    Args:
        model (AudioTransformer): Trained model.
        seed_spectrogram (torch.Tensor): Seed spectrogram to start generation.
        max_length (int): Maximum length of the generated spectrogram.
    
    Returns:
        generated_spectrogram (np.ndarray): Generated spectrogram.
    """
    with torch.no_grad():
        # Initialize the input with the seed spectrogram
        input_spectrogram = seed_spectrogram.unsqueeze(0)  # Add batch dimension
        
        # Generate the spectrogram step by step
        for _ in range(max_length - seed_spectrogram.shape[0]):
            # Predict the next time step
            output = model(input_spectrogram, input_spectrogram)
            next_step = output[:, -1:, :]  # Take the last predicted time step
            
            # Append the predicted step to the input
            input_spectrogram = torch.cat([input_spectrogram, next_step], dim=1)
        
        # Remove the batch dimension and convert to numpy array
        generated_spectrogram = input_spectrogram.squeeze(0).cpu().numpy()
    
    return generated_spectrogram

def spectrogram_to_audio(spectrogram, sample_rate=16000, n_fft=1024, hop_length=256):
    """
    Convert a spectrogram back to an audio waveform.
    
    Args:
        spectrogram (np.ndarray): Generated spectrogram.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
    
    Returns:
        audio (np.ndarray): Reconstructed audio waveform.
    """
    # Convert the spectrogram from dB to power
    spectrogram = librosa.db_to_power(spectrogram)
    
    # Use Griffin-Lim algorithm to reconstruct the audio
    audio = librosa.feature.inverse.mel_to_audio(
        spectrogram,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    
    return audio

def generate_audio(config):
    """
    Generate audio using the trained model.
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters and paths.
    
    Returns:
        audio (np.ndarray): Generated audio waveform.
    """
    # Extract hyperparameters
    input_dim = config["input_dim"]
    model_dim = config["model_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    max_length = config["max_length"]
    sample_rate = config["sample_rate"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    checkpoint_path = config["checkpoint_best_path"]
    seed_spectrogram_path = config["seed_spectrogram_path"]

    # Load the trained model
    model = load_model(checkpoint_path, input_dim, model_dim, num_heads, num_layers, max_length)
    print("Model loaded successfully.")

    # Load the seed spectrogram
    seed_spectrogram = torch.load(seed_spectrogram_path)
    print(f"Seed spectrogram shape: {seed_spectrogram.shape}")

    # Generate a spectrogram
    generated_spectrogram = generate_spectrogram(model, seed_spectrogram, max_length)
    print(f"Generated spectrogram shape: {generated_spectrogram.shape}")

    # Convert the spectrogram to audio
    audio = spectrogram_to_audio(generated_spectrogram, sample_rate, n_fft, hop_length)
    print(f"Generated audio length: {len(audio)} samples")

    return audio