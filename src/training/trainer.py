import time
import math
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from ..models.bilstm import build_pyramidal_bilstm
from ..data.dataset import AudioDataset
from ..utils.logger import log_model_summary, log_print as print
from ..utils.metrics import comprehensive_evaluation
from ..config import device

def train_and_evaluate_model(num_pyramid_layers, X_train, y_train, X_val, y_val, X_test, y_test,
                            base_units=128, learning_rate=1e-4, weight_decay=1e-5,
                            batch_size=8, epochs=100, patience=7):
    """
    Train and evaluate a pyramidal BiLSTM with specified number of pyramid layers
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL WITH {num_pyramid_layers} PYRAMID LAYER(S)")
    print(f"{'='*60}")
   
    try:
        wandb_run = wandb.init(
            project="deepfake-pyramidal-bilstm",
            config={
                "num_pyramid_layers": num_pyramid_layers,
                "base_units": base_units,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "n_mels": 80,
                "feature_type": "HPSS_Triple_Channel",
                "feature_dim": 240,
                "margin": 2
            },
            reinit=True
        )
    except Exception as e:
        print(f"Wandb init failed: {e}")
        wandb_run = None
   
    model, optimizer, criterion = build_pyramidal_bilstm(
        (X_train.shape[1], X_train.shape[2]),
        base_units=base_units,
        num_pyramid_layers=num_pyramid_layers,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
   
    log_model_summary(model, f"Pyramidal BiLSTM ({num_pyramid_layers} layers)")
   
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
    def onecycle_lambda(epoch):
        max_lr_factor = 5.0
        min_lr_factor = 0.01
        warmup_epochs = 5
        peak_epochs = 15
        total_epochs = epochs
       
        if epoch < warmup_epochs:
            return 1.0 + (max_lr_factor - 1.0) * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + peak_epochs:
            return max_lr_factor
        else:
            remaining_epochs = total_epochs - warmup_epochs - peak_epochs
            progress = (epoch - warmup_epochs - peak_epochs) / remaining_epochs
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (max_lr_factor - min_lr_factor) * cosine_factor
   
    scheduler = LambdaLR(optimizer, lr_lambda=onecycle_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-8)
   
    timestamp = int(time.time())
    best_checkpoint_path = f"best_model_{num_pyramid_layers}layers_{timestamp}.pt"
    last_checkpoint_path = f"last_epoch_model_{num_pyramid_layers}layers_{timestamp}.pt"
   
    print(f"\n{'='*80}")
    print(f"ENHANCED TRAINING PIPELINE - PYTORCH")
    print(f"{'='*80}")
    print(f"Optimizer: AdamW (weight_decay={weight_decay}, lr={learning_rate})")
    print(f"Learning Rate: OneCycle (Base: {learning_rate}, Peak: {learning_rate*5}, Min: {learning_rate*0.01})")
    print(f"Schedule: Warmup (5 epochs) → Peak (15 epochs) → Cosine Annealing")
    print(f"Max Epochs: {epochs} (with early stopping, patience={patience})")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"\n💾 CHECKPOINT STRATEGY:")
    print(f"  • Best model: Saved when validation loss improves")
    print(f"  • Last epoch: Saved at training completion")
    print(f"  • Evaluation: Uses best model weights")
    print(f"{'='*80}")
   
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
   
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
       
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.float().to(device)
           
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
           
            train_loss += loss.item() * batch_features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (predictions == batch_labels.long()).sum().item()
            train_total += batch_labels.size(0)
       
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
       
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
       
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.float().to(device)
               
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
               
                val_loss += loss.item() * batch_features.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (predictions == batch_labels.long()).sum().item()
                val_total += batch_labels.size(0)
       
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
       
        scheduler.step()
        plateau_scheduler.step(val_loss)
       
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
       
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"LR: {current_lr:.2e}")
             
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr
            })
       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_checkpoint_path)
            print(f"💾 Best model saved: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
   
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'history': history
    }, last_checkpoint_path)
    print(f"💾 Last epoch checkpoint saved: {last_checkpoint_path}")
   
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
   
    print(f"Loading best model from checkpoint for evaluation...")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
   
    print(f"\nGenerating predictions for comprehensive evaluation...")
    model.eval()
   
    with torch.no_grad():
        y_test_prob = []
        for batch_features, _ in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_test_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_test_prob = np.array(y_test_prob)
       
        y_val_prob = []
        for batch_features, _ in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_val_prob.extend(probs if probs.ndim > 0 else [probs.item()])
        y_val_prob = np.array(y_val_prob)
   
    test_metrics = comprehensive_evaluation(y_test, y_test_prob, dataset_name="Test")
    val_metrics = comprehensive_evaluation(y_val, y_val_prob, dataset_name="Validation")
   
    total_params = sum(p.numel() for p in model.parameters())
   
    results = {
        'pyramid_layers': num_pyramid_layers,
        'total_params': total_params,
        'epochs_trained': len(history['train_loss']),
        'best_epoch': np.argmin(history['val_loss']) + 1,
        'best_val_loss': min(history['val_loss']),
        'best_val_acc': max(history['val_acc']),
        'test_metrics': test_metrics,
        'val_metrics': val_metrics,
        'checkpoint_path': best_checkpoint_path,
        'best_checkpoint_path': best_checkpoint_path,
        'last_checkpoint_path': last_checkpoint_path,
        'history': history,
        'evaluation_type': 'comprehensive_with_bootstrap'
    }
   
    print(f"\nFINAL PERFORMANCE SUMMARY (using best model):")
    print(f"  Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Test EER: {test_metrics['eer']:.4f} @ threshold={test_metrics['eer_threshold']:.4f}")
   
    if wandb_run is not None:
        wandb.log({
            "test_auc_roc": test_metrics['auc_roc'],
            "test_f1_score": test_metrics['f1_score'],
            "test_balanced_accuracy": test_metrics['balanced_accuracy'],
            "test_eer": test_metrics['eer']
        })
        wandb.finish()
       
    return results, model

def train_with_optimized_pipeline(model, optimizer, criterion, loader_info,
                                  epochs=100, patience=7):
    """
    Train model using optimized DataLoader pipeline.
    PyTorch equivalent of TF.Data training.
    """
    print(f"\n{'='*80}")
    print(f"TRAINING WITH OPTIMIZED DATALOADER PIPELINE - PYTORCH")
    print(f"{'='*80}")
   
    train_loader = loader_info['train_loader']
    val_loader = loader_info['val_loader']
   
    try:
        wandb_run = wandb.init(
            project="deepfake-pyramidal-bilstm",
            name="Optimized_DataLoader_Pipeline",
            config={
                "epochs": epochs,
                "patience": patience,
                "batch_size": loader_info.get('batch_size', 8),
                "n_mels": 80,
                "feature_type": "HPSS_Triple_Channel",
                "feature_dim": 240,
                "margin": 2,
                "pipeline": "optimized_dataloader"
            },
            reinit=True
        )
    except Exception as e:
        print(f"Wandb init failed: {e}")
        wandb_run = None
   
    def onecycle_lambda(epoch):
        max_lr_factor = 5.0
        min_lr_factor = 0.01
        warmup_epochs = 5
        peak_epochs = 15
       
        if epoch < warmup_epochs:
            return 1.0 + (max_lr_factor - 1.0) * (epoch / warmup_epochs)
        elif epoch < warmup_epochs + peak_epochs:
            return max_lr_factor
        else:
            remaining_epochs = epochs - warmup_epochs - peak_epochs
            progress = (epoch - warmup_epochs - peak_epochs) / remaining_epochs
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (max_lr_factor - min_lr_factor) * cosine_factor
   
    scheduler = LambdaLR(optimizer, lr_lambda=onecycle_lambda)
   
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
   
    timestamp = int(time.time())
    best_checkpoint_path = f"dataloader_best_model_{timestamp}.pt"
    last_checkpoint_path = f"dataloader_last_epoch_{timestamp}.pt"
   
    print(f"Training Configuration:")
    print(f"  Max Epochs: {epochs}")
    print(f"  Early Stopping Patience: {patience}")
    print(f"  Best Checkpoint: {best_checkpoint_path}")
    print(f"  Last Checkpoint: {last_checkpoint_path}")
   
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
       
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.float().to(device)
           
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
           
            train_loss += loss.item() * batch_features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (predictions == batch_labels.long()).sum().item()
            train_total += batch_labels.size(0)
       
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
       
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
       
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.float().to(device)
               
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
               
                val_loss += loss.item() * batch_features.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (predictions == batch_labels.long()).sum().item()
                val_total += batch_labels.size(0)
       
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
       
        scheduler.step()
       
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
       
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"LR: {current_lr:.2e}")
             
        if wandb_run is not None:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr
            })
       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_checkpoint_path)
            print(f"💾 Best model saved: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
   
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'history': history
    }, last_checkpoint_path)
    print(f"💾 Last epoch checkpoint saved: {last_checkpoint_path}")
   
    print(f"\n📁 SAVED CHECKPOINTS:")
    print(f"  Best model: {best_checkpoint_path}")
    print(f"  Last epoch: {last_checkpoint_path}")
   
    return {
        'history': history,
        'checkpoint_path': best_checkpoint_path,
        'best_checkpoint_path': best_checkpoint_path,
        'last_checkpoint_path': last_checkpoint_path,
        'epochs_trained': len(history['train_loss']),
        'best_val_loss': min(history['val_loss'])
    }
