# utils/audio_processing.py

import librosa
import torch
import numpy as np
from pathlib import Path
import torchaudio


def load_audio(file_path, sample_rate=16000, target_duration=5.0):
    """
    Load a WAV file, resample it, and trim/pad it to a fixed duration.
    
    Args:
        file_path (str): Path to the WAV file.
        sample_rate (int): Target sample rate.
        target_duration (float): Target duration in seconds.
    
    Returns:
        audio (np.ndarray): Audio signal with fixed duration.
    """
    audio, _ = librosa.load(file_path, sr=sample_rate) 

    if target_duration is not None: 
        target_length = int(target_duration * sample_rate) 
        if len(audio) > target_length: 
            audio = audio[:target_length] 
        else: 
            audio = np.pad(audio, (0, max(0, target_length - len(audio))), mode='constant') 
    return audio 

def normalize_audio(audio): 
    """ 
    Normalize audio to have zero mean and unit variance.  
    
    Args: 
        audio (np.ndarray): Audio data. 
    
    Returns: 
        audio (np.ndarray): Normalized audio. 
    """ 
    audio = audio - np.mean(audio) 
    audio = audio / np.std(audio) 
    return audio 

def extract_mel_spectrogram_and_normalize(audio, sample_rate, n_mels, n_fft, hop_length): 
    """
    Extract Mel-spectrogram from audio.
    
    Args:
        audio (np.ndarray): Audio data.
        sample_rate (int): Sample rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
    
    Returns:
        spectrogram (torch.Tensor): Mel-spectrogram.
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,  # Make sure to use n_fft, not n_mels here
        hop_length=hop_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0
    ) 
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max) 
    
    spectrogram = torch.from_numpy(spectrogram).float() 
    return spectrogram 


def preprocess_audio(file_path, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=256, target_duration=5.0):
    """
    Preprocess a WAV file: load, normalize, and extract Mel-spectrogram.
    
    Args:
        file_path (str): Path to the WAV file.
        sample_rate (int): Target sample rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        target_duration (float): Target duration in seconds.
    
    Returns:
        spectrogram (torch.Tensor): Preprocessed Mel-spectrogram. 
    """ 
    # Load and resample audio 
    print('load_audio ', file_path) 
    audio = load_audio(file_path, sample_rate, target_duration=None) 
    print(audio.shape) 
    
    # Calculate the number of samples per segment 
    print('split the audio into segement') 

    segment_length = int(target_duration * sample_rate) 
    overlap_length = int(1 * sample_rate)  # 1-second overlap in samples 

    # Split the audio into overlapping segments 
    segments = [audio[i:i + segment_length] for i in range(0, len(audio) - segment_length + 1, segment_length - overlap_length)] 

    print('In total', len(segments), 'segments') 

    # Process each segment
    spectrograms = []
    for idx, segment in enumerate(segments):
        # print(idx,segment.shape)
        # Normalize aio
        if not np.isfinite(segment).all():
            print(f"Is infinite: {np.isfinite(segment).all()}")
            continue
        # segment = normalize_audio(segment)
        # print('Extract Mel-spec',segment.shape) 
        # Extract Mel-spectrogram 
        try: 
            spectrogram = extract_mel_spectrogram_and_normalize(segment, sample_rate, n_mels, n_fft, hop_length)
        except Exception as e:
            print(f"Error in segment {idx}: {e}")


        spectrogram = (spectrogram + 80) / 80
        spectrogram = spectrogram.transpose(0, 1)

        # spectrogram = spectrogram/mel_spectram_norm # Normalize the spectrogram
        print(spectrogram.min(), spectrogram.max(),spectrogram.mean(),spectrogram.shape)
        
        spectrograms.append(spectrogram)
    
    return spectrograms
