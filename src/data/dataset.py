import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from .features import extract_mel_features_with_augment

class AudioDataset(Dataset):
    """PyTorch Dataset for pre-extracted audio features"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
   
    def __len__(self):
        return len(self.labels)
   
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AudioDatasetFromFiles(Dataset):
    """
    PyTorch Dataset that loads audio files on-the-fly with augmentation.
    """
    def __init__(self, file_paths, labels, max_seconds=2.0, apply_augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.max_seconds = max_seconds
        self.apply_augment = apply_augment
   
    def __len__(self):
        return len(self.file_paths)
   
    def __getitem__(self, idx):
        features = extract_mel_features_with_augment(
            self.file_paths[idx],
            max_seconds=self.max_seconds,
            apply_augment=self.apply_augment
        )
       
        if features is None:
            # Return zero features if extraction fails (80 is N_MELS)
            features = np.zeros((int(self.max_seconds * 16000 // 256) + 1, 80), dtype=np.float32)
       
        label = self.labels[idx]
        return torch.FloatTensor(features), torch.LongTensor([label]).squeeze()

def collate_fn_variable_length(batch):
    """
    Custom collate function to pad variable-length sequences.
    """
    features, labels = zip(*batch)
   
    # Find max length in batch
    max_len = max([f.size(0) for f in features])
   
    # Pad sequences
    padded_features = []
    for f in features:
        pad_amount = max_len - f.size(0)
        padded = F.pad(f, (0, 0, 0, pad_amount), value=0.0)
        padded_features.append(padded)
   
    return torch.stack(padded_features), torch.stack([torch.tensor(l) for l in labels])

def pad_sequences_numpy(X_list, maxlen, mask_value=0.0):
    """
    Pad sequences to uniform length using explicit mask value.
    """
    n_samples = len(X_list)
    n_features = X_list[0].shape[1]
   
    X_padded = np.full((n_samples, maxlen, n_features), mask_value, dtype=np.float32)
   
    for i, x in enumerate(X_list):
        length = min(x.shape[0], maxlen)
        X_padded[i, :length, :] = x[:length, :]
   
    return X_padded

def create_optimized_dataloaders(dataset_path, batch_size=8, num_workers=4,
                                max_seconds=2.0, apply_scaler=True, dataset_type='for-rerec'):
    import os
    from .features import get_files_and_labels
    print(f"\n{'='*80}")
    print(f"CREATING OPTIMIZED PYTORCH DATALOADERS ({dataset_type.upper()})")
    print(f"{'='*80}")
   
    if dataset_type == 'asvspoof':
        # Removed as requested, this will just fallback
        pass
        
    train_audio_dir = os.path.join(dataset_path, 'training')
    train_files, train_labels = get_files_and_labels(train_audio_dir)
    
    val_audio_dir = os.path.join(dataset_path, 'validation')
    val_files, val_labels = get_files_and_labels(val_audio_dir)
    
    test_audio_dir = os.path.join(dataset_path, 'testing')
    test_files, test_labels = get_files_and_labels(test_audio_dir)
   
    train_real_files = [f for i, f in enumerate(train_files) if train_labels[i] == 0]
    train_fake_files = [f for i, f in enumerate(train_files) if train_labels[i] == 1]
    val_real_files = [f for i, f in enumerate(val_files) if val_labels[i] == 0]
    val_fake_files = [f for i, f in enumerate(val_files) if val_labels[i] == 1]
    test_real_files = [f for i, f in enumerate(test_files) if test_labels[i] == 0]
    test_fake_files = [f for i, f in enumerate(test_files) if test_labels[i] == 1]
   
    print(f"Dataset Statistics:")
    print(f"  Training: {len(train_files)} files ({len(train_real_files)} real, {len(train_fake_files)} fake)")
    print(f"  Validation: {len(val_files)} files ({len(val_real_files)} real, {len(val_fake_files)} fake)")
    print(f"  Testing: {len(test_files)} files ({len(test_real_files)} real, {len(test_fake_files)} fake)")
   
    # Create datasets
    train_dataset = AudioDatasetFromFiles(train_files, train_labels, max_seconds, apply_augment=True)
    val_dataset = AudioDatasetFromFiles(val_files, val_labels, max_seconds, apply_augment=False)
    test_dataset = AudioDatasetFromFiles(test_files, test_labels, max_seconds, apply_augment=False)
   
    # Create DataLoaders with optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_variable_length,
        persistent_workers=True if num_workers > 0 else False
    )
   
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_variable_length,
        persistent_workers=True if num_workers > 0 else False
    )
   
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_variable_length,
        persistent_workers=True if num_workers > 0 else False
    )
   
    print(f"\nDataLoader Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Num Workers: {num_workers} (parallel data loading)")
    print(f"  Pin Memory: {torch.cuda.is_available()} (GPU optimization)")
    print(f"  Training Augmentation: ON")
    print(f"  Val/Test Augmentation: OFF")
   
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_size': len(train_files),
        'val_size': len(val_files),
        'test_size': len(test_files),
        'batch_size': batch_size
    }
