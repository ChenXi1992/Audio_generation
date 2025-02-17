# utils/preprocess_data.py

import os
import torch
from tqdm import tqdm
from utils.audio_processing import preprocess_audio  # Import the function


def preprocess_dataset(raw_data_dir, processed_data_dir, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=256, target_duration=5.0, max_length=313, mel_spectram_norm=100):
    """ 
    Preprocess all WAV files in the raw data directory and save the processed data.
    
    Args:
        raw_data_dir (str): Directory containing raw WAV files.
        processed_data_dir (str): Directory to save processed Mel-spectrograms.
        sample_rate (int): Target sample rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        target_duration (float): Target duration in seconds.
        max_length: The maximum length of the spectrogram 
    """
    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Get list of WAV files in the raw data directory
    file_paths = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith(".wav")]

    # Preprocess each file and save the Mel-spectrogram
    for file_path in tqdm(file_paths, desc="Preprocessing audio files"):

        # Preprocess the audio file
        spectrograms= preprocess_audio(file_path, sample_rate, n_mels, n_fft, hop_length, target_duration) 
        print("In total has ",len(spectrograms),'segments') 
        # Save each segment's Mel-spectrogram  
        base_name = os.path.basename(file_path).replace(".wav", "") 
        for i, spectrogram in enumerate(spectrograms): 
            # print(spectrogram.min(), spectrogram.max(),spectrogram.shape)
            file_name = f"{base_name}_segment_{i}.pt"
            save_path = os.path.join(processed_data_dir, file_name)
            torch.save(spectrogram, save_path) 
    print(f"Preprocessing complete. Processed files saved to {processed_data_dir}")



