# Audio Deepfake Detection - Pyramidal BiLSTM

This repository contains a modular, production-ready PyTorch implementation for Audio Deepfake Detection using a custom **Pyramidal BiLSTM** architecture with attention pooling and extensive augmentation.

## Features
- **Modular Design:** Clear separation of configuration, data loading, feature extraction, models, and training loops.
- **Advanced Architecture:** Pyramidal BiLSTM that downsamples sequences iteratively, coupled with a self-attention pooling mechanism.
- **Comprehensive Augmentation:** On-the-fly audio augmentation (Time-stretching, Pitch-shifting, Gaussian Noise) and SpecAugment (Time & Frequency masking) for Mel-Spectrograms.
- **Optimized Pipeline:** Custom PyTorch `DataLoaders` with multi-threading, dynamic batching, and an efficient `.npz` feature caching system.
- **Experiment Tracking:** Integrated with Weights & Biases (W&B) for live dashboard tracking.

## Repository Structure

```text
├── main.py                     # Primary entry point for training and evaluation
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # Global seeds, device configuration
│   ├── data/
│   │   ├── augment.py          # Data augmentation routines
│   │   ├── dataset.py          # PyTorch Datasets and DataLoaders
│   │   └── features.py         # Mel-Spectrogram extraction and caching
│   ├── models/
│   │   ├── bilstm.py           # Core Pyramidal BiLSTM structure
│   │   └── components.py       # Custom PyramidalDownsample and Attention layers
│   ├── training/
│   │   ├── cross_val.py        # Stratified K-Fold CV logic
│   │   └── trainer.py          # Training loops and WandB logging
│   └── utils/
│       ├── logger.py           # Formatted logging system
│       └── metrics.py          # EER, ROC-AUC, and confidence intervals
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd btp
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv1
   venv1\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place your dataset inside the project directory (e.g., `for-rerec/` or `ASVspoof/`) and adjust the paths in `main.py`.

Run the training pipeline:
```bash
python main.py
```

### Feature Caching
To speed up subsequent training runs, extracted Mel-Spectrograms are automatically cached to disk. If you change your augmentation parameters, force a fresh extraction by running:
```bash
python main.py --fresh-features
```

## Logging
Logs are printed to the console and automatically saved to the `logs/` directory.

## License
MIT License
