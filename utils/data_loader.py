# utils/data_loader.py
import os
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, data_dir, max_length=1000):
        """
        Args:
            data_dir (str): Directory containing processed .pt files.
            max_length (int): Maximum sequence length (time steps).
        """
        self.data_dir = data_dir
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.max_length = max_length

        # Check if there are any files in the directory
        if len(self.file_paths) == 0:
            raise ValueError(f"No .pt files found in {data_dir}. Please check the preprocessing step.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        spectrogram = torch.load(file_path)  # Shape: (n_mels, time_steps)
        # print("loading spectgrogram shape",spectrogram.shape)
        return spectrogram 


def create_data_loader(data_dir, batch_size=4, shuffle=True, max_length=1000):
    """
    Create a DataLoader for audio data.
    
    Args:
        data_dir (str): Directory containing processed .pt files.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        max_length (int): Maximum sequence length (time steps).
    
    Returns:
        data_loader (DataLoader): PyTorch DataLoader.
    """
    dataset = AudioDataset(data_dir, max_length)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset