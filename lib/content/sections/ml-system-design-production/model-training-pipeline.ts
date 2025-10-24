export const modelTrainingPipeline = {
  title: 'Model Training Pipeline',
  id: 'model-training-pipeline',
  content: `
# Model Training Pipeline

## Introduction

A robust **model training pipeline** automates the entire training process—from data loading to model evaluation. In production, you'll retrain models regularly (daily, weekly, monthly) as new data arrives.

**Manual training doesn't scale**:
- Inconsistent processes
- Human errors
- No automation
- Slow iteration

**Automated pipeline benefits**:
- Reproducible training
- Scheduled retraining
- Parallel experiments
- Monitoring and alerts

This section covers building production-grade training pipelines for scalable, automated model development.

### Training Pipeline Components

\`\`\`
Data Loading → Preprocessing → Feature Engineering → Training → Evaluation → Model Storage
     ↓              ↓                 ↓                ↓            ↓             ↓
  Validation    Transform        Feature Store      MLflow    Metrics Log    Registry
\`\`\`

By the end of this section, you'll understand:
- Building automated training pipelines
- Distributed training for large models
- Hyperparameter optimization at scale
- Training monitoring and checkpointing
- GPU utilization and mixed precision training

---

## Training Pipeline Architecture

### Complete Training Pipeline

\`\`\`python
"""
Production Training Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline
    
    Handles:
    - Data loading and validation
    - Preprocessing
    - Training
    - Evaluation
    - Model storage
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Step 1: Load data from source
        """
        logger.info("Loading data...")
        
        data_path = self.config.get('data_path')
        
        # In production: load from S3, database, etc.
        # For demo: generate synthetic data
        n_samples = self.config.get('n_samples', 10000)
        n_features = self.config.get('n_features', 20)
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        y = pd.Series(
            X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1
        )
        
        logger.info(f"✓ Loaded {len(X)} samples, {len(X.columns)} features")
        
        return X, y
    
    def validate_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Step 2: Validate data quality
        """
        logger.info("Validating data...")
        
        # Check for missing values
        missing = X.isnull().sum().sum()
        if missing > 0:
            raise ValueError(f"Found {missing} missing values")
        
        # Check for infinite values
        if np.isinf(X.values).any():
            raise ValueError("Found infinite values")
        
        # Check target
        if y.isnull().any():
            raise ValueError("Target contains missing values")
        
        logger.info("✓ Data validation passed")
    
    def preprocess(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 3: Preprocess features
        """
        logger.info("Preprocessing features...")
        
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✓ Features scaled")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series
    ):
        """
        Step 4: Train model
        """
        logger.info("Training model...")
        
        model_type = self.config.get('model_type', 'RandomForest')
        
        if model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**self.config['model_params'])
        elif model_type == 'XGBoost':
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**self.config['model_params'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        logger.info("✓ Model trained")
        
        return self.model
    
    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Step 5: Evaluate model
        """
        logger.info("Evaluating model...")
        
        # Predictions
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        test_preds = self.model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'train_r2': r2_score(y_train, train_preds),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_preds)),
            'val_r2': r2_score(y_val, val_preds),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
            'test_r2': r2_score(y_test, test_preds)
        }
        
        logger.info(f"✓ Evaluation complete")
        logger.info(f"  Val RMSE: {metrics['val_rmse']:.6f}")
        logger.info(f"  Test RMSE: {metrics['test_rmse']:.6f}")
        
        return metrics
    
    def save_model(self, model_path: str):
        """
        Step 6: Save model
        """
        logger.info(f"Saving model to {model_path}...")
        
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config
        }, model_path)
        
        logger.info("✓ Model saved")
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete pipeline
        """
        logger.info("\\n" + "="*60)
        logger.info("Starting Training Pipeline")
        logger.info("="*60 + "\\n")
        
        # Start MLflow run
        with mlflow.start_run(run_name=self.config.get('run_name', 'training_run')):
            
            # Log config
            mlflow.log_params(self.config['model_params'])
            
            # 1. Load data
            X, y = self.load_data()
            
            # 2. Validate
            self.validate_data(X, y)
            
            # 3. Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42  # 60/20/20 split
            )
            
            logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            
            # 4. Preprocess
            X_train_scaled, X_val_scaled, X_test_scaled = self.preprocess(
                X_train, X_val, X_test
            )
            
            # 5. Train
            model = self.train(X_train_scaled, y_train, X_val_scaled, y_val)
            
            # 6. Evaluate
            metrics = self.evaluate(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test
            )
            
            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # 7. Save model
            model_path = self.config.get('model_path', 'model.pkl')
            self.save_model(model_path)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("\\n" + "="*60)
            logger.info("Training Pipeline Complete")
            logger.info("="*60 + "\\n")
            
            return {
                'metrics': metrics,
                'model_path': model_path
            }


# Example configuration
config = {
    'run_name': 'trading_model_v1',
    'n_samples': 10000,
    'n_features': 20,
    'model_type': 'RandomForest',
    'model_params': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    },
    'model_path': 'trading_model.pkl'
}

# Run pipeline
# pipeline = TrainingPipeline(config)
# result = pipeline.run()
# print(f"\\nFinal metrics: {result['metrics']}")
\`\`\`

---

## Distributed Training

### Data Parallelism with PyTorch

\`\`\`python
"""
Distributed Training with PyTorch
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import os

def setup_distributed():
    """
    Setup distributed training environment
    """
    # Initialize process group
    dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    return rank, world_size, device


def cleanup_distributed():
    """
    Cleanup distributed training
    """
    dist.destroy_process_group()


class SimpleNN(nn.Module):
    """
    Simple neural network for demo
    """
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def train_distributed(rank, world_size):
    """
    Distributed training function
    
    Run with:
    torchrun --nproc_per_node=4 train_script.py
    """
    # Setup
    device = torch.device(f'cuda:{rank}')
    
    # Create model and move to device
    model = SimpleNN(input_dim=20).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create dataset
    n_samples = 10000
    X = torch.randn(n_samples, 20)
    y = torch.randn(n_samples, 1)
    
    dataset = TensorDataset(X, y)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    
    for epoch in range(10):
        # Set epoch for sampler (important for shuffling)
        sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            # Move to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Average loss across all processes
        avg_loss = epoch_loss / len(dataloader)
        
        if rank == 0:  # Only print from main process
            print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.6f}")
    
    # Cleanup
    cleanup_distributed()


# To run:
# torchrun --nproc_per_node=4 script.py

print("Distributed training example defined")
print("Run with: torchrun --nproc_per_node=NUM_GPUS script.py")
\`\`\`

### Model Parallelism

\`\`\`python
"""
Model Parallelism for Large Models
"""

import torch
import torch.nn as nn

class LargeModelParallel(nn.Module):
    """
    Split large model across multiple GPUs
    
    Use when:
    - Model too large for single GPU
    - Example: GPT-3 has 175B parameters
    """
    
    def __init__(self):
        super().__init__()
        
        # First half on GPU 0
        self.layer1 = nn.Linear(1000, 5000).to('cuda:0')
        self.layer2 = nn.Linear(5000, 5000).to('cuda:0')
        
        # Second half on GPU 1
        self.layer3 = nn.Linear(5000, 5000).to('cuda:1')
        self.layer4 = nn.Linear(5000, 100).to('cuda:1')
    
    def forward(self, x):
        # Start on GPU 0
        x = x.to('cuda:0')
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        
        # Move to GPU 1
        x = x.to('cuda:1')
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        
        return x


# Example usage
if torch.cuda.device_count() >= 2:
    model = LargeModelParallel()
    
    # Input on CPU, will be moved in forward pass
    x = torch.randn(32, 1000)
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
else:
    print("Need at least 2 GPUs for model parallelism")
\`\`\`

---

## Hyperparameter Optimization at Scale

### Ray Tune for Distributed HPO

\`\`\`python
"""
Hyperparameter Optimization with Ray Tune
"""

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_split import train_test_split

def train_model_ray(config):
    """
    Training function for Ray Tune
    
    Args:
        config: Dictionary of hyperparameters
    """
    # Generate data
    np.random.seed(42)
    X = np.random.randn(5000, 20)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(5000) * 0.1
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with hyperparameters from config
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        random_state=42,
        n_jobs=1  # Important: let Ray handle parallelism
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    val_preds = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    
    # Report metrics to Ray Tune
    tune.report(val_rmse=val_rmse)


def run_hyperparameter_optimization():
    """
    Run distributed hyperparameter optimization
    """
    # Define search space
    search_space = {
        'n_estimators': tune.choice([100, 200, 300, 500]),
        'max_depth': tune.randint(3, 20),
        'min_samples_split': tune.randint(2, 20),
        'min_samples_leaf': tune.randint(1, 10)
    }
    
    # ASHA scheduler (early stopping)
    scheduler = ASHAScheduler(
        metric='val_rmse',
        mode='min',
        max_t=100,  # Max iterations
        grace_period=10,  # Min iterations before stopping
        reduction_factor=2
    )
    
    # Optuna search algorithm
    search_alg = OptunaSearch()
    
    # Run tuning
    analysis = tune.run(
        train_model_ray,
        config=search_space,
        num_samples=50,  # Number of trials
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={'cpu': 2, 'gpu': 0},  # Resources per trial
        verbose=1
    )
    
    # Get best hyperparameters
    best_config = analysis.get_best_config(metric='val_rmse', mode='min')
    
    print(f"\\n🏆 Best hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    print(f"\\nBest val_rmse: {analysis.best_result['val_rmse']:.6f}")
    
    return best_config, analysis


# Run optimization
# best_config, analysis = run_hyperparameter_optimization()
\`\`\`

### Optuna for Sequential HPO

\`\`\`python
"""
Hyperparameter Optimization with Optuna
"""

import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def objective(trial):
    """
    Objective function for Optuna
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Validation RMSE
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(5000, 20)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(5000) * 0.1
    
    # Train model
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    
    # Cross-validation
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=1  # Parallel CV folds
    )
    
    # Return negative RMSE (Optuna minimizes)
    return -scores.mean()


def run_optuna_optimization(n_trials=50):
    """
    Run Optuna optimization
    """
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),  # Tree-structured Parzen Estimator
        pruner=optuna.pruners.MedianPruner()  # Prune unpromising trials
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, n_jobs=4)  # Parallel trials
    
    # Results
    print(f"\\n=== Optuna Results ===")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.6f}")
    
    print(f"\\n🏆 Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Importance
    importances = optuna.importance.get_param_importances(study)
    
    print(f"\\nParameter importances:")
    for key, value in importances.items():
        print(f"  {key}: {value:.4f}")
    
    return study.best_params, study


# Run optimization
# best_params, study = run_optuna_optimization(n_trials=50)
\`\`\`

---

## Training Monitoring and Checkpointing

### Model Checkpointing

\`\`\`python
"""
Model Checkpointing for Long Training
"""

import torch
import torch.nn as nn
from pathlib import Path

class CheckpointManager:
    """
    Manage model checkpoints during training
    """
    
    def __init__(self, checkpoint_dir='checkpoints', keep_best_n=3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.checkpoints = []  # List of (metric, path) tuples
        self.best_metric = float('inf')
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        metrics,
        is_best=False
    ):
        """
        Save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
        torch.save(checkpoint, epoch_path)
        
        # Save best
        val_loss = metrics.get('val_loss', float('inf'))
        
        if val_loss < self.best_metric or is_best:
            self.best_metric = val_loss
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            
            print(f"✓ New best model: val_loss={val_loss:.6f}")
        
        # Keep only best N checkpoints
        self.checkpoints.append((val_loss, epoch_path))
        self.checkpoints.sort(key=lambda x: x[0])  # Sort by metric
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_best_n:
            for _, path in self.checkpoints[self.keep_best_n:]:
                if path.exists() and path != latest_path:
                    path.unlink()
            
            self.checkpoints = self.checkpoints[:self.keep_best_n]
    
    def load_checkpoint(self, model, optimizer, checkpoint_path='best.pth'):
        """
        Load checkpoint
        """
        path = self.checkpoint_dir / checkpoint_path
        
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return None
        
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        
        print(f"✓ Loaded checkpoint from epoch {epoch}")
        
        return epoch, metrics


# Example usage
model = nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters())

checkpoint_mgr = CheckpointManager()

# Simulate training
for epoch in range(10):
    # Train...
    val_loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
    
    metrics = {'val_loss': val_loss}
    
    checkpoint_mgr.save_checkpoint(
        model, optimizer, epoch, metrics
    )
    
    print(f"Epoch {epoch}: val_loss={val_loss:.6f}")

print(f"\\nBest model saved: {checkpoint_mgr.best_metric:.6f}")
\`\`\`

### TensorBoard Integration

\`\`\`python
"""
Training Monitoring with TensorBoard
"""

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np

def train_with_tensorboard(model, train_loader, val_loader, epochs=10):
    """
    Training with TensorBoard logging
    """
    # Create writer
    writer = SummaryWriter('runs/training_experiment')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    global_step = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log training loss
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            global_step += 1
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch metrics
        writer.add_scalars('Loss/epoch', {
            'train': avg_train_loss,
            'val': avg_val_loss
        }, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log model weights histogram
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
    
    writer.close()
    
    print("\\nView results with: tensorboard --logdir=runs")


# Example
# model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
# train_with_tensorboard(model, train_loader, val_loader)
\`\`\`

---

## GPU Optimization

### Mixed Precision Training

\`\`\`python
"""
Mixed Precision Training for Faster Training
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, epochs=10):
    """
    Training with automatic mixed precision (AMP)
    
    Benefits:
    - 2-3x faster training
    - 2x less memory
    - Minimal accuracy loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Create GradScaler
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")


# Example
if torch.cuda.is_available():
    model = nn.Sequential(
        nn.Linear(100, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 1)
    )
    
    # Create dummy data
    train_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 100),
        torch.randn(1000, 1)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    # train_with_mixed_precision(model, train_loader)
    print("Mixed precision training example ready")
else:
    print("CUDA not available")
\`\`\`

---

## Key Takeaways

1. **Automated Pipelines**: Reproducible, scheduled, monitored
2. **Distributed Training**: Scale to multiple GPUs with DDP
3. **HPO at Scale**: Ray Tune, Optuna for parallel hyperparameter search
4. **Checkpointing**: Save best models, resume training
5. **Monitoring**: TensorBoard for real-time tracking
6. **GPU Optimization**: Mixed precision for 2-3x speedup

**Trading-Specific**:
- Walk-forward validation for time series
- Retrain weekly/monthly as new data arrives
- Track trading metrics (Sharpe, drawdown) during training
- A/B test model versions in paper trading

**Next Steps**: With training pipelines automated, we'll cover model serving and deployment in the next section.
`,
};
