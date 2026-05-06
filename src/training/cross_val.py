import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .trainer import train_and_evaluate_model
from ..data.dataset import pad_sequences_numpy, AudioDataset
from ..utils.logger import log_print as print
from ..config import device
from ..models.bilstm import build_pyramidal_bilstm
from torch.utils.data import DataLoader
from ..utils.metrics import comprehensive_evaluation

def stratified_kfold_cv(X_data, y_data, X_test, y_test, num_pyramid_layers=2,
                       k_folds=5, random_state=42, base_units=128,
                       learning_rate=1e-4, weight_decay=1e-5, batch_size=8):
    print(f"\n{'='*90}")
    print(f"STRATIFIED {k_folds}-FOLD CROSS-VALIDATION ANALYSIS - PYTORCH")
    print(f"{'='*90}")
   
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    fold_results = []
   
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        print(f"\n{'='*80}")
        print(f"TRAINING FOLD {fold_idx + 1}/{k_folds}")
        print(f"{'='*80}")
       
        X_train_fold = [X_data[i] for i in train_idx]
        y_train_fold = y_data[train_idx]
        X_val_fold = [X_data[i] for i in val_idx]
        y_val_fold = y_data[val_idx]
       
        fold_lengths = []
        fold_lengths.extend([x.shape[0] for x in X_train_fold])
        fold_lengths.extend([x.shape[0] for x in X_val_fold])
        fold_max_T = max(fold_lengths)
       
        X_train_padded = pad_sequences_numpy(X_train_fold, fold_max_T)
        X_val_padded = pad_sequences_numpy(X_val_fold, fold_max_T)
       
        scaler_fold = StandardScaler()
        train_lengths = [x.shape[0] for x in X_train_fold]
        non_padded_frames = []
        for i, length in enumerate(train_lengths):
            non_padded_frames.append(X_train_padded[i, :length, :])
        non_padded_data = np.vstack(non_padded_frames)
        scaler_fold.fit(non_padded_data)
       
        N_train, T_train, n_features_fold = X_train_padded.shape
        X_train_scaled = scaler_fold.transform(X_train_padded.reshape(-1, n_features_fold)).reshape(N_train, T_train, n_features_fold)
        X_val_scaled = scaler_fold.transform(X_val_padded.reshape(-1, n_features_fold)).reshape(X_val_padded.shape[0], X_val_padded.shape[1], n_features_fold)
       
        try:
            fold_result, fold_model = train_and_evaluate_model(
                num_pyramid_layers, X_train_scaled, y_train_fold,
                X_val_scaled, y_val_fold, X_val_scaled, y_val_fold,
                base_units=base_units, learning_rate=learning_rate,
                weight_decay=weight_decay, batch_size=batch_size,
                epochs=100, patience=7
            )
           
            fold_result['fold_idx'] = fold_idx + 1
            fold_result['train_size'] = len(X_train_fold)
            fold_result['val_size'] = len(X_val_fold)
            fold_result['scaler'] = scaler_fold
            fold_result['fold_max_T'] = fold_max_T
           
            fold_results.append(fold_result)
           
            print(f"\nFold {fold_idx + 1} Completed Successfully")
        except Exception as e:
            print(f"\nERROR in Fold {fold_idx + 1}: {str(e)}")
            continue
       
        del fold_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
   
    if len(fold_results) == 0:
        print("ERROR: No folds completed successfully!")
        return None
   
    cv_stats = analyze_cv_results(fold_results, k_folds)
   
    best_fold_idx = np.argmax([result['test_metrics']['auc_roc'] for result in fold_results])
    best_result = fold_results[best_fold_idx]
   
    best_fold_max_T = best_result['fold_max_T']
    X_test_padded = pad_sequences_numpy(X_test, best_fold_max_T)
    best_scaler = best_result['scaler']
    X_test_scaled = best_scaler.transform(X_test_padded.reshape(-1, X_test_padded.shape[2])).reshape(X_test_padded.shape)
   
    try:
        checkpoint = torch.load(best_result['checkpoint_path'])
        best_model, _, _ = build_pyramidal_bilstm(
            (X_test_scaled.shape[1], X_test_scaled.shape[2]),
            base_units=base_units,
            num_pyramid_layers=num_pyramid_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.eval()
       
        test_dataset = AudioDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
       
        y_test_prob = []
        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(device)
                outputs = best_model(batch_features).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                y_test_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_test_prob = np.array(y_test_prob)
       
        test_metrics_final = comprehensive_evaluation(y_test, y_test_prob, dataset_name="Hold-out Test")
        del best_model
    except Exception as e:
        print(f"Error loading best model: {e}")
        test_metrics_final = None
   
    return {
        'cv_type': 'stratified_kfold',
        'k_folds': k_folds,
        'successful_folds': len(fold_results),
        'fold_results': fold_results,
        'cv_statistics': cv_stats,
        'best_fold_idx': best_fold_idx + 1,
        'best_fold_result': best_result,
        'holdout_test_metrics': test_metrics_final,
        'model_config': {'pyramid_layers': num_pyramid_layers, 'random_state': random_state}
    }

def analyze_cv_results(fold_results, k_folds):
    metrics_keys = ['auc_roc', 'auc_pr', 'f1_score', 'balanced_accuracy', 'eer', 'precision', 'recall']
    fold_metrics = {key: [r['test_metrics'][key] for r in fold_results] for key in metrics_keys}
   
    cv_stats = {}
    for key in metrics_keys:
        values = np.array(fold_metrics[key])
        cv_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
   
    return cv_stats
