import numpy as np
import librosa

def normalize_audio(audio):
    """
    Normalize audio to the range [-1, 1].
    
    Args:
        audio (np.ndarray): Raw audio waveform.
    
    Returns:
        normalized_audio (np.ndarray): Normalized audio waveform.
    """
    return audio / np.max(np.abs(audio))

def reduce_noise(audio, sample_rate=16000, n_fft=1024, hop_length=256):
    """
    Reduce noise in the audio using a spectral gate.
    
    Args:
        audio (np.ndarray): Raw audio waveform.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
    
    Returns:
        denoised_audio (np.ndarray): Denoised audio waveform.
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)

    # Apply spectral gate
    threshold = np.median(magnitude) * 0.1  # Adjust threshold as needed
    magnitude[magnitude < threshold] = 0

    # Reconstruct audio
    stft_denoised = magnitude * phase
    denoised_audio = librosa.istft(stft_denoised, hop_length=hop_length)
    return denoised_audio

def post_process_audio(audio, sample_rate=16000):
    """
    Post-process generated audio (normalize and reduce noise).
    
    Args:
        audio (np.ndarray): Raw audio waveform.
        sample_rate (int): Sample rate of the audio.
    
    Returns:
        processed_audio (np.ndarray): Post-processed audio waveform.
    """
    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sample_rate)
    return audio