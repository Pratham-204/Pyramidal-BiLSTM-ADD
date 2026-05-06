import os
import sys
import numpy as np
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm
from .augment import apply_audio_augmentation, spec_augment
from ..config import FEATURE_CACHE_DIR

FORCE_FRESH_FEATURES = '--fresh-features' in sys.argv

def get_files_and_labels(audio_dir):
    """
    Parse a standard directory structure:
    audio_dir/
      real/
      fake/
    """
    files = []
    labels = []
    
    if isinstance(audio_dir, list):
        audio_dir = audio_dir[0]
        
    real_dir = os.path.join(audio_dir, 'real')
    fake_dir = os.path.join(audio_dir, 'fake')
    
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.endswith('.wav') or f.endswith('.flac'):
                files.append(os.path.join(real_dir, f))
                labels.append(0)  # 0 for real
                
    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir):
            if f.endswith('.wav') or f.endswith('.flac'):
                files.append(os.path.join(fake_dir, f))
                labels.append(1)  # 1 for spoof
                
    return files, labels

def extract_mel_features_with_augment(file_path, max_seconds=2.0, n_mels=80, n_fft=1024,
                                     hop_length=256, apply_augment=False):
    """
    Extract log-Mel spectrogram features with comprehensive augmentation pipeline.
    """
    try:
        y, sr = librosa.load(file_path, duration=max_seconds, sr=16000, mono=True)
        if y.size == 0:
            return None
    except Exception:
        return None
   
    if apply_augment:
        y = apply_audio_augmentation(y, sr, augment_prob=0.6)
   
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256, fmax=sr//2)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max).T
       
    if apply_augment:
        log_mel = spec_augment(
            log_mel,
            time_mask_param=15,
            freq_mask_param=10,
            n_time_masks=1,
            n_freq_masks=1
        )
   
    return log_mel

def load_split(protocol_file, audio_dir, apply_augment=False):
    from ..config import N_MELS
    X, y = [], []
    augment_str = "FULL AUGMENTATION (Audio + Spectral)" if apply_augment else "NO AUGMENTATION"
    print(f"Loading data from {audio_dir}")
   
    files, labels = get_files_and_labels(audio_dir)
    print(f"  Found {len(files)} files")
   
    def _process_file(i, file_path):
        arr = extract_mel_features_with_augment(
            file_path,
            max_seconds=2.0,
            n_mels=N_MELS,
            apply_augment=apply_augment
        )
        return arr, labels[i]
       
    print("Parallel processing to speed up Mel-Spectrogram extraction...")
    results = Parallel(n_jobs=16, return_as="generator")(
        delayed(_process_file)(i, fp) for i, fp in enumerate(tqdm(files, desc="Extracting features"))
    )
   
    for arr, lbl in results:
        if arr is not None:
            X.append(arr)
            y.append(lbl)
       
    print(f"Total: {len(X)} samples with {augment_str.lower()}")
    return X, np.array(y, dtype=np.int64)

def save_features_to_cache(X_list, y_array, cache_name):
    cache_path = os.path.join(FEATURE_CACHE_DIR, f'{cache_name}.npz')
    save_dict = {'y': y_array, 'n_samples': len(X_list)}
    for i, x in enumerate(X_list):
        save_dict[f'x_{i}'] = x
   
    np.savez_compressed(cache_path, **save_dict)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  💾 Cached {len(X_list)} samples to {cache_path} ({size_mb:.1f} MB)")

def load_features_from_cache(cache_name):
    cache_path = os.path.join(FEATURE_CACHE_DIR, f'{cache_name}.npz')
    if not os.path.exists(cache_path):
        return None
   
    print(f"  ⚡ Loading cached features from {cache_path}...")
    data = np.load(cache_path, allow_pickle=False)
    n_samples = int(data['n_samples'])
    y_array = data['y']
    X_list = [data[f'x_{i}'] for i in range(n_samples)]
   
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  ✅ Loaded {n_samples} samples from cache ({size_mb:.1f} MB)")
    return X_list, y_array

def load_split_with_cache(protocol_file, audio_dir, apply_augment=False, cache_name=None):
    if cache_name and not FORCE_FRESH_FEATURES:
        cached = load_features_from_cache(cache_name)
        if cached is not None:
            return cached
   
    if FORCE_FRESH_FEATURES and cache_name:
        print(f"  🔄 --fresh-features flag set, re-extracting...")
   
    X_list, y_array = load_split(protocol_file, audio_dir, apply_augment=apply_augment)
   
    if cache_name:
        save_features_to_cache(X_list, y_array, cache_name)
   
    return X_list, y_array
