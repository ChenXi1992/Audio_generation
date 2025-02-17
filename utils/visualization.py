import matplotlib.pyplot as plt
import librosa.display

def plot_spectrogram(spectrogram, title="Spectrogram"):
    """
    Plot a Mel-spectrogram.
    
    Args:
        spectrogram (np.ndarray): Mel-spectrogram of shape (n_mels, time_steps).
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis="time", y_axis="mel", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.show()

def plot_waveform(audio, sample_rate=16000, title="Waveform"):
    """
    Plot an audio waveform.
    
    Args:
        audio (np.ndarray): Raw audio waveform.
        sample_rate (int): Sample rate of the audio.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sample_rate)
    plt.title(title)
    plt.show()

def plot_loss_curves(train_losses, val_losses):
    """
    Plot training, validation, and test loss curves.
    
    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()