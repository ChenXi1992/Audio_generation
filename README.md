Sound Generation
This is an experimental project aimed at generating natural sounds using deep learning.

Current Capabilities
- The model can currently generate 4-second audio clips.


Getting Started
Data Preparation: Store .wav files in the directory: data/raw
Running the Project: Model Training & Execution: Run main.ipynb
Audio Generation: Run audio_generation.ipynb

Current Challenges
The model output tends to be too static and lacks temporal variation when generating natural sounds like water, crickets, etc.

To-Do List
1. Improve Loss Functions
Contrastive Loss: Enforce variability by adding a loss function between adjacent time frames.
Multi-Scale Spectrogram Loss: Capture fine details in the audio.
Autocorrelation Loss: Experiment with different loss functions to better model the natural temporal dependencies in sound.
2. Enhance Audio Reconstruction
Train a Mel-to-Audio Network: Current conversion from Mel-spectrogram to audio using Librosa introduces robotic artifacts in certain sounds like water.
3. Implement a Better Sampling Strategy
Develop a strategy to improve the diversity and realism of generated samples.