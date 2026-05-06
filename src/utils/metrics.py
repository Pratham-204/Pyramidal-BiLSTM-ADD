import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, brier_score_loss)
from .logger import log_print as print

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for any metric.
    """
    n_samples = len(y_true)
    bootstrap_scores = []
   
    # Set random seed for reproducibility
    np.random.seed(42)
   
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
       
        # Calculate metric on bootstrap sample
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        except:
            # Skip invalid bootstrap samples
            continue
   
    bootstrap_scores = np.array(bootstrap_scores)
   
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
   
    return {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'ci_lower': np.percentile(bootstrap_scores, lower_percentile),
        'ci_upper': np.percentile(bootstrap_scores, upper_percentile),
        'confidence': confidence
    }

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Calculate calibration metrics for probability predictions.
    """
    brier_score = brier_score_loss(y_true, y_prob)
   
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
   
    ece = 0
    mce = 0
   
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
       
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
           
            bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
   
    return {
        'brier_score': brier_score,
        'ece': ece,
        'mce': mce
    }

def comprehensive_evaluation(y_true, y_prob, dataset_name="Test"):
    if len(np.unique(y_true)) < 2:
        print(f"\n--- {dataset_name} has only 1 class (dummy labels). Skipping metrics calculation ---")
        return {'auc_roc': 0.0, 'f1_score': 0.0, 'balanced_accuracy': 0.0, 'eer': 0.5, 'auc_roc_ci': {'ci_lower': 0, 'ci_upper': 0}, 'f1_ci': {'ci_lower': 0, 'ci_upper': 0}, 'eer_threshold': 0.5}

    y_pred = (y_prob > 0.5).astype(int)
   
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION METRICS - {dataset_name.upper()} SET")
    print(f"{'='*80}")
   
    accuracy = np.mean(y_true == y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    eer, eer_threshold = calculate_eer(y_true, y_prob)
   
    auc_pr = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
   
    calibration = calculate_calibration_metrics(y_true, y_prob)
   
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
   
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall
    balanced_accuracy = (sensitivity + specificity) / 2
   
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (95% CI):")
    print(f"Computing bootstrap estimates with 1000 samples...")
   
    auc_roc_ci = bootstrap_metric(y_true, y_prob,
                                  lambda yt, yp: roc_auc_score(yt, yp), n_bootstrap=1000)
    auc_pr_ci = bootstrap_metric(y_true, y_prob,
                                 lambda yt, yp: average_precision_score(yt, yp), n_bootstrap=1000)
    f1_ci = bootstrap_metric(y_true, y_pred,
                             lambda yt, yp: f1_score(yt, yp), n_bootstrap=1000)
   
    print(f"\nCORE PERFORMANCE METRICS:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f} [{auc_roc_ci['ci_lower']:.4f}, {auc_roc_ci['ci_upper']:.4f}]")
    print(f"  AUC-PR: {auc_pr:.4f} [{auc_pr_ci['ci_lower']:.4f}, {auc_pr_ci['ci_upper']:.4f}]")
    print(f"  F1-Score: {f1:.4f} [{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]")
    print(f"  EER: {eer:.4f} @ threshold={eer_threshold:.4f}")
   
    print(f"\nPER-CLASS METRICS:")
    print(f"  Precision (Fake Detection): {precision:.4f}")
    print(f"  Recall/Sensitivity (Fake Detection): {recall:.4f}")
    print(f"  Specificity (Real Detection): {specificity:.4f}")
   
    print(f"\nCONFUSION MATRIX:")
    print(f"  True Negatives (Real→Real): {tn}")
    print(f"  False Positives (Real→Fake): {fp}")
    print(f"  False Negatives (Fake→Real): {fn}")
    print(f"  True Positives (Fake→Fake): {tp}")
   
    print(f"\nCALIBRATION ANALYSIS:")
    print(f"  Brier Score: {calibration['brier_score']:.4f} (lower is better)")
    print(f"  Expected Calibration Error (ECE): {calibration['ece']:.4f}")
    print(f"  Maximum Calibration Error (MCE): {calibration['mce']:.4f}")
   
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'auc_roc': auc_roc,
        'auc_roc_ci': auc_roc_ci,
        'auc_pr': auc_pr,
        'auc_pr_ci': auc_pr_ci,
        'f1_score': f1,
        'f1_ci': f1_ci,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'confusion_matrix': cm,
        'calibration': calibration,
        'bootstrap_confidence': 0.95
    }
