
## Setup Environment

You will need Python 3.10 or greater, as well as Poetry.

### Poetry Installation

1. Install pipx:
   **Mac:**
   ```sh
   brew install pipx && pipx ensurepath
   ```
   **Windows:**
   
   First, install Scoop:
   ```sh
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
   ```
   
   Then, install pipx:
   ```sh
   scoop install pipx
   pipx ensurepath
   ```

2. Install Poetry:
   ```sh
   pipx install poetry
   ```

### Installing Python Environment and Packages
```sh
poetry install
```

### Adding packages to poetry environment
All packages are held in the the `pyproject.toml` file, this includes their versioning.
To add packages:
```sh
poetry add <package_name>
```

## Model Configuration and Training

### Model Configuration

The model architecture can be customized through the `ModelConfig` class:

```python
model_config = ModelConfig(
    num_classes=2,          # Number of output classes
    model_name='resnet18',  # Backbone architecture ['resnet18', 'resnet34', 'resnet50']
    pretrained=True,        # Use pretrained weights
    dropout_rate=0.1,       # Dropout rate for regularization
    hidden_size=512,        # Size of hidden layer in classifier
    freeze_layers=2,        # Number of layers to freeze in backbone
    activation="relu"       # Activation function ['relu', 'gelu', 'silu']
)
```

### Training Configuration

Training parameters can be adjusted using the `TrainingConfig` class:

```python
training_config = TrainingConfig(
    batch_size=32,          # Batch size for training
    num_epochs=100,         # Maximum number of epochs
    base_lr=1e-4,          # Base learning rate
    weight_decay=0.01,      # L2 regularization
    label_smoothing=0.1,    # Label smoothing factor
    warmup_epochs=2,        # Number of warmup epochs
    patience=10,            # Early stopping patience
    num_folds=5,           # Number of cross-validation folds
    test_size=0.2,         # Proportion of data for testing
    mixup_alpha=0.2,       # Mixup augmentation alpha
    grad_clip=1,           # Gradient clipping value
    seed=42                # Random seed for reproducibility
)
```

## Running the Model

### Training

1. Basic training with default parameters:
```python
python algorithm.py
```

2. Models are automatically saved under the `models/` directory:
```
models/{model_type}/model_best.pth
```

### Evaluation

To evaluate a trained model:

```python
python algorithm.py --eval-only
```

## Dataset Stuff
Update the data paths in `algorithm.py`:
```python
data_path = "/path/to/your/dataset"
healthy_path = os.path.join(data_path, "Healthy Areas")
cancer_path = os.path.join(data_path, "Cancer & Fibrosis Areas")
```