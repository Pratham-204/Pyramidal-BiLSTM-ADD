import logging
import os
import sys
from datetime import datetime
import torch
import numpy as np
import librosa

def setup_logging():
    """Set up comprehensive logging to both console and file."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/pyramidal_bilstm_pytorch_training_{timestamp}.log"
   
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
   
    logger = logging.getLogger('pyramidal_bilstm_pytorch')
   
    logger.info("="*80)
    logger.info("PYRAMIDAL BiLSTM AUDIO DEEPFAKE DETECTION SYSTEM - PYTORCH")
    logger.info("="*80)
    logger.info(f"Log file created: {log_filename}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Librosa version: {librosa.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        logger.info(f"Using CUDA device: {device_id}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(device_id)}")
    logger.info("="*80)
   
    return logger, log_filename

logger, current_log_file = setup_logging()

original_print = print
def log_print(*args, **kwargs):
    """Enhanced print function that logs to both console and file."""
    message = ' '.join(str(arg) for arg in args)
    logger.info(message)
    original_print(*args, **kwargs)

def get_logger():
    return logger

def log_model_summary(model, model_name="Pyramidal BiLSTM"):
    """Log detailed model summary and architecture information."""
    log_print(f"\n{'='*60}")
    log_print(f"MODEL SUMMARY: {model_name}")
    log_print(f"{'='*60}")
    log_print(model)
   
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
   
    log_print(f"\nModel Parameters:")
    log_print(f"  Total parameters: {total_params:,}")
    log_print(f"  Trainable parameters: {trainable_params:,}")
    log_print(f"  Non-trainable parameters: {non_trainable_params:,}")
    log_print(f"{'='*60}")

def log_experiment_results(results, experiment_name):
    """Log comprehensive experiment results."""
    log_print(f"\n{'='*80}")
    log_print(f"EXPERIMENT RESULTS: {experiment_name}")
    log_print(f"{'='*80}")
   
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, (int, float)):
                log_print(f"  {key}: {value:.4f}")
            elif isinstance(value, str):
                log_print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    log_print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
                else:
                    log_print(f"  {key}: {value}")
            else:
                log_print(f"  {key}: {value}")
    else:
        log_print(f"Results: {results}")
   
    log_print(f"{'='*80}")
