import numpy as np
import librosa

def spec_augment(mel_spec, time_mask_param=15, freq_mask_param=10, n_time_masks=2, n_freq_masks=2):
    """
    Apply SpecAugment to log-mel spectrogram for data augmentation.
   
    Args:
        mel_spec: (T, F) mel spectrogram
        time_mask_param: Maximum width of time mask
        freq_mask_param: Maximum width of frequency mask  
        n_time_masks: Number of time masks to apply
        n_freq_masks: Number of frequency masks to apply
   
    Returns:
        Augmented mel spectrogram with same shape (T, F)
    """
    mel_spec = mel_spec.copy()
    T, n_freq_bins = mel_spec.shape
   
    for _ in range(n_freq_masks):
        if freq_mask_param > 0 and n_freq_bins > freq_mask_param:
            mask_width = np.random.randint(0, min(freq_mask_param, n_freq_bins))
            if mask_width > 0:
                mask_start = np.random.randint(0, n_freq_bins - mask_width)
                mel_spec[:, mask_start:mask_start + mask_width] = 0
   
    for _ in range(n_time_masks):
        if time_mask_param > 0 and T > time_mask_param:
            mask_width = np.random.randint(0, min(time_mask_param, T))
            if mask_width > 0:
                mask_start = np.random.randint(0, T - mask_width)
                mel_spec[mask_start:mask_start + mask_width, :] = 0
               
    return mel_spec

def add_colored_noise(audio, noise_type='white', snr_db=15):
    """
    Add colored noise at specified SNR.
    """
    signal_power = np.mean(audio ** 2)
    # Handle silence/zeros to avoid division by zero
    if signal_power == 0:
        return audio
        
    noise_power = signal_power / (10 ** (snr_db / 10))
   
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    elif noise_type == 'pink':
        white_noise = np.random.normal(0, 1, len(audio))
        noise = np.convolve(white_noise, [1, -0.5], mode='same')
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
    elif noise_type == 'brown':
        white_noise = np.random.normal(0, 1, len(audio))
        noise = np.cumsum(white_noise) * 0.02
        noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
   
    return audio + noise

def speed_change(audio, sr, speed_factor):
    """Change speed of audio without changing pitch."""
    if speed_factor == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def pitch_shift(audio, sr, n_steps):
    """Shift pitch by n_steps semitones."""
    if n_steps == 0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def apply_audio_augmentation(audio, sr, augment_prob=0.7):
    """
    Apply random audio augmentations to raw audio signal.
    """
    augmented = audio.copy()
   
    # Apply speed perturbation (70% chance)
    if np.random.random() < augment_prob:
        speed_factors = [0.9, 1.0, 1.1]
        speed_factor = np.random.choice(speed_factors)
        if speed_factor != 1.0:
            augmented = speed_change(augmented, sr, speed_factor)
   
    # Apply pitch shifting (50% chance, smaller range)
    if np.random.random() < (augment_prob * 0.7):
        pitch_steps = [-1, 0, 1]
        n_steps = np.random.choice(pitch_steps)
        if n_steps != 0:
            augmented = pitch_shift(augmented, sr, n_steps)
   
    # Apply noise addition (80% chance)
    if np.random.random() < (augment_prob + 0.1):
        noise_type = np.random.choice(['white', 'pink'])
        snr_db = np.random.uniform(10, 20)
        augmented = add_colored_noise(augmented, noise_type, snr_db)
   
    # Normalize to prevent clipping
    max_val = np.max(np.abs(augmented))
    if max_val > 1.0:
        augmented = augmented / max_val * 0.95
       
    return augmented
