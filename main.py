import os
import sys
import codecs

# Force UTF-8 encoding for stdout/stderr to fix Windows emoji printing issues
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import set_seed, DATASET_PATH
from src.utils.logger import log_experiment_results, log_print as print
from src.data.features import load_split_with_cache, FORCE_FRESH_FEATURES
from src.data.dataset import pad_sequences_numpy, create_optimized_dataloaders
from src.training.trainer import train_and_evaluate_model, train_with_optimized_pipeline
from src.training.cross_val import stratified_kfold_cv

def main():
    set_seed(42)
    print("\n" + "="*90)
    print("ULTRA-ENHANCED PYRAMIDAL BiLSTM WITH COMPREHENSIVE AUGMENTATION - PYTORCH")
    print("Testing 2 pyramid layers + attention + multi-level data augmentation")
    print("="*90)
   
    # Load data with COMPREHENSIVE augmentation pipeline ONLY for training set
    print("="*80)
    print("LOADING DATASET WITH COMPREHENSIVE AUGMENTATION PIPELINE")
    print("="*80)
    if FORCE_FRESH_FEATURES:
        print("⚠️  --fresh-features flag detected: will re-extract all features from audio")
    else:
        print("💡 Using feature cache if available (use --fresh-features to force re-extraction)")

    # Using for-rerec dataset instead
    REREC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'for-rerec', 'for-rerecorded')
    
    train_protocol = None
    train_audio_dir = os.path.join(REREC_PATH, 'training')
    
    dev_protocol = None
    dev_audio_dir = os.path.join(REREC_PATH, 'validation')
    
    eval_protocol = None
    eval_audio_dir = os.path.join(REREC_PATH, 'testing')

    X_train_list, y_train = load_split_with_cache(train_protocol, train_audio_dir, apply_augment=True, cache_name='rerec_train_logmel_augmented')
    X_val_list,   y_val   = load_split_with_cache(dev_protocol, dev_audio_dir, apply_augment=False, cache_name='rerec_val_logmel')
    X_test_list,  y_test  = load_split_with_cache(eval_protocol, eval_audio_dir, apply_augment=False, cache_name='rerec_test_logmel')

    all_lengths = []
    if X_train_list: all_lengths.extend([x.shape[0] for x in X_train_list])
    if X_val_list: all_lengths.extend([x.shape[0] for x in X_val_list])
    if X_test_list: all_lengths.extend([x.shape[0] for x in X_test_list])
    
    if not all_lengths:
        print("Warning: No audio data loaded. Please check your DATASET_PATH and protocol paths.")
        return

    global_max_T = max(all_lengths)
    
    X_train = pad_sequences_numpy(X_train_list, global_max_T)
    X_val = pad_sequences_numpy(X_val_list, global_max_T)
    X_test = pad_sequences_numpy(X_test_list, global_max_T)
    
    scaler = StandardScaler()
    N, T, n_features = X_train.shape
    
    train_lengths = [x.shape[0] for x in X_train_list]
    non_padded_frames = []
    for i, length in enumerate(train_lengths):
        non_padded_frames.append(X_train[i, :length, :])
    non_padded_data = np.vstack(non_padded_frames)
    scaler.fit(non_padded_data)
    
    X_train = scaler.transform(X_train.reshape(-1, n_features)).reshape(N, T, n_features)
    X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], X_val.shape[1], n_features)
    X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], X_test.shape[1], n_features)
    
    print(f"\n{'='*90}")
    print("APPROACH 1: SINGLE TRAIN/VALIDATION/TEST SPLIT - PYTORCH")
    print(f"{'='*90}")
    single_split_results, single_model = train_and_evaluate_model(
        2, X_train, y_train, X_val, y_val, X_test, y_test,
        base_units=128, learning_rate=1e-4, weight_decay=1e-5,
        batch_size=8, epochs=100, patience=7
    )
    log_experiment_results(single_split_results, "Single Split Validation - PyTorch")
    
    del single_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Approach 2: Cross Validation
    X_combined = list(X_train_list) + list(X_val_list)
    y_combined = np.concatenate([y_train, y_val])
    cv_results = stratified_kfold_cv(
        X_combined, y_combined, X_test_list, y_test,
        num_pyramid_layers=2, k_folds=5,
        base_units=128, learning_rate=1e-4, batch_size=8
    )
    if cv_results:
        log_experiment_results(cv_results, "Cross Validation Results - PyTorch")

    # Optuna has been removed per request
    
    # Approach 4: Optimized Dataloader Pipeline
    print(f"\n{'='*90}")
    print("APPROACH 4: ON-THE-FLY FEATURE EXTRACTION & AUGMENTATION - PYTORCH")
    print(f"{'='*90}")
    loader_info = create_optimized_dataloaders(
        dataset_path=REREC_PATH,
        batch_size=8,
        num_workers=4,
        max_seconds=2.0,
        dataset_type='for-rerec'
    )
    
    from src.models.bilstm import build_pyramidal_bilstm
    # We just need to build the model structure
    model, optimizer, criterion = build_pyramidal_bilstm(
        input_shape=(80, 240), # arbitrary since it's just used to init the model
        base_units=128,
        num_pyramid_layers=2,
        learning_rate=1e-4
    )
    
    dl_results = train_with_optimized_pipeline(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loader_info=loader_info,
        epochs=100,
        patience=7
    )
    
if __name__ == "__main__":
    main()
