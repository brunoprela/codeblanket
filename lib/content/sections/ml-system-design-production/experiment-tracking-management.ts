export const experimentTrackingManagement = {
  title: 'Experiment Tracking & Management',
  id: 'experiment-tracking-management',
  content: `
# Experiment Tracking & Management

## Introduction

**"If you can't measure it, you can't improve it."**

In machine learning, **experiments are everything**. A single project might involve hundreds of experiments:
- Different models (XGBoost vs Neural Network)
- Different hyperparameters (learning rate, depth)
- Different features (50 vs 100 features)
- Different data (train on 1 year vs 5 years)

**Without proper tracking, you'll**:
- Lose track of what worked
- Waste time repeating failed experiments
- Can't reproduce results
- Can't collaborate effectively

This section covers building robust experiment tracking systems using tools like **MLflow**, **Weights & Biases**, and **Neptune.ai**.

### What to Track

\`\`\`python
"""
Complete Experiment Information
"""

experiment_info = {
    "metadata": {
        "experiment_id": "exp_2024_01_15_001",
        "timestamp": "2024-01-15T10:30:00",
        "user": "data_scientist_1",
        "project": "trading_strategy",
        "description": "Test XGBoost with 100 technical indicators"
    },
    
    "data": {
        "training_data": "s3://data/train_2020_2023.parquet",
        "validation_data": "s3://data/val_2024_q1.parquet",
        "data_version": "v1.2.3",
        "data_hash": "a1b2c3d4...",
        "num_samples": 1_000_000,
        "features": ["sma_20", "rsi", "macd", ...],  # 100 features
        "target": "returns_next_5min"
    },
    
    "model": {
        "algorithm": "XGBoost",
        "framework": "xgboost==1.7.0",
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8
        },
        "code_version": "commit_abc123"
    },
    
    "training": {
        "duration_seconds": 3600,
        "hardware": "AWS p3.2xlarge (V100 GPU)",
        "cost_usd": 3.06,
        "iterations": 500,
        "early_stopping": True,
        "best_iteration": 387
    },
    
    "metrics": {
        "train": {
            "rmse": 0.0023,
            "mae": 0.0018,
            "r2": 0.67
        },
        "validation": {
            "rmse": 0.0029,
            "mae": 0.0022,
            "r2": 0.54
        },
        "test": {
            "rmse": 0.0031,
            "mae": 0.0024,
            "r2": 0.51
        },
        "trading": {
            "sharpe_ratio": 1.85,
            "max_drawdown": 0.12,
            "win_rate": 0.56
        }
    },
    
    "artifacts": {
        "model_path": "s3://models/exp_001/model.pkl",
        "plots": ["feature_importance.png", "learning_curve.png"],
        "logs": "s3://logs/exp_001/",
        "predictions": "s3://predictions/exp_001/"
    }
}
\`\`\`

By the end of this section, you'll understand:
- Experiment tracking tools (MLflow, W&B, Neptune)
- Versioning code, data, and models
- Comparing experiments systematically
- Reproducibility best practices
- Collaboration workflows

---

## MLflow: Getting Started

### Basic MLflow Setup

\`\`\`python
"""
MLflow Basic Example
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Set tracking URI (local or remote server)
mlflow.set_tracking_uri("http://localhost:5000")  # Or file:///path/to/mlruns

# Set experiment
mlflow.set_experiment("trading_strategy_development")

def train_model_with_mlflow(X, y, params):
    """
    Train model with complete MLflow tracking
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run (run_name=f"rf_depth_{params['max_depth']}"):
        
        # 1. Log parameters
        mlflow.log_params (params)
        
        # 2. Log dataset info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("num_features", X_train.shape[1])
        
        # 3. Train model
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        # 5. Log metrics
        train_rmse = np.sqrt (mean_squared_error (y_train, train_preds))
        val_rmse = np.sqrt (mean_squared_error (y_val, val_preds))
        train_r2 = r2_score (y_train, train_preds)
        val_r2 = r2_score (y_val, val_preds)
        
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_r2", val_r2)
        
        # 6. Log model
        mlflow.sklearn.log_model (model, "model")
        
        # 7. Log artifacts (plots, feature importance)
        import matplotlib.pyplot as plt
        
        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort (importances)[::-1][:10]
        
        plt.figure (figsize=(10, 6))
        plt.bar (range(10), importances[indices])
        plt.title('Top 10 Feature Importances')
        plt.savefig('feature_importance.png')
        plt.close()
        
        mlflow.log_artifact('feature_importance.png')
        
        # 8. Log tags for organization
        mlflow.set_tags({
            "model_type": "RandomForest",
            "use_case": "trading",
            "data_version": "v1.0",
            "stage": "development"
        })
        
        print(f"\\n✓ Run logged: val_rmse={val_rmse:.6f}, val_r2={val_r2:.4f}")
        
        return model, val_rmse, val_r2


# Generate sample data
np.random.seed(42)
n_samples = 10000
n_features = 20

X = np.random.randn (n_samples, n_features)
y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn (n_samples) * 0.1

# Experiment with different hyperparameters
param_grid = [
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10},
    {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": 15, "min_samples_split": 2}
]

results = []
for i, params in enumerate (param_grid):
    print(f"\\n=== Experiment {i+1}/{len (param_grid)} ===")
    model, val_rmse, val_r2 = train_model_with_mlflow(X, y, params)
    results.append({
        "params": params,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

# Find best model
best = min (results, key=lambda x: x['val_rmse'])
print(f"\\n🏆 Best model: {best['params']}")
print(f"   Val RMSE: {best['val_rmse']:.6f}")
print(f"   Val R2: {best['val_r2']:.4f}")
\`\`\`

### MLflow UI and Model Registry

\`\`\`python
"""
MLflow Model Registry for Production
"""

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

def register_best_model (experiment_name, metric="val_rmse"):
    """
    Find best run and register to Model Registry
    """
    # Get experiment
    experiment = client.get_experiment_by_name (experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found")
        return
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    
    print(f"\\nBest run: {run_id}")
    print(f"  {metric}: {best_run.data.metrics[metric]:.6f}")
    
    # Register model
    model_uri = f"runs:/{run_id}/model"
    model_name = "trading_strategy_model"
    
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    print(f"\\n✓ Model registered: {model_name} version {model_version.version}")
    
    return model_version


def promote_model_to_production (model_name, version):
    """
    Promote model from Staging to Production
    """
    client = MlflowClient()
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"✓ Model {model_name} v{version} promoted to Production")


def load_production_model (model_name):
    """
    Load latest production model
    """
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model (model_uri)
    
    print(f"✓ Loaded production model: {model_name}")
    
    return model


# Example usage
# model_version = register_best_model("trading_strategy_development")
# promote_model_to_production("trading_strategy_model", model_version.version)
# production_model = load_production_model("trading_strategy_model")
\`\`\`

---

## Weights & Biases (W&B)

### W&B for Advanced Tracking

\`\`\`python
"""
Weights & Biases Example
"""

import wandb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_with_wandb(X, y, config):
    """
    Train with W&B tracking
    
    W&B advantages:
    - Beautiful visualizations
    - Real-time monitoring
    - Hyperparameter sweeps
    - Collaboration features
    """
    # Initialize W&B run
    run = wandb.init(
        project="trading-strategy",
        name=f"rf_depth_{config['max_depth']}",
        config=config,
        tags=["random_forest", "v1"],
        notes="Experiment with technical indicators"
    )
    
    # Access config
    config = wandb.config
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_rmse = np.sqrt (mean_squared_error (y_train, train_preds))
    val_rmse = np.sqrt (mean_squared_error (y_val, val_preds))
    train_r2 = r2_score (y_train, train_preds)
    val_r2 = r2_score (y_val, val_preds)
    
    # Log metrics
    wandb.log({
        "train/rmse": train_rmse,
        "train/r2": train_r2,
        "val/rmse": val_rmse,
        "val/r2": val_r2
    })
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X.shape[1])],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create W&B Table
    table = wandb.Table (dataframe=feature_importance.head(10))
    wandb.log({"feature_importance": table})
    
    # Log plots
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Predictions vs Actual
    axes[0].scatter (y_val, val_preds, alpha=0.5)
    axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Predictions vs Actual')
    
    # Residuals
    residuals = y_val - val_preds
    axes[1].hist (residuals, bins=50, edgecolor='black')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    
    plt.tight_layout()
    
    # Log figure
    wandb.log({"predictions_plot": wandb.Image (fig)})
    plt.close()
    
    # Log model
    wandb.log_artifact (model, name="trading_model", type="model")
    
    # Finish run
    run.finish()
    
    print(f"✓ W&B run complete: {run.url}")
    
    return model


# Example usage
config = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "learning_rate": 0.01
}

# Generate sample data
np.random.seed(42)
X = np.random.randn(10000, 20)
y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(10000) * 0.1

# Train with W&B (comment out if not authenticated)
# model = train_with_wandb(X, y, config)
\`\`\`

### W&B Hyperparameter Sweeps

\`\`\`python
"""
Automated Hyperparameter Sweeps with W&B
"""

import wandb

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val/rmse',
        'goal': 'minimize'
    },
    'parameters': {
        'n_estimators': {
            'values': [100, 200, 300, 500]
        },
        'max_depth': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 20
        },
        'min_samples_split': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 20
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
            'max': 0.1
        }
    }
}

def train_sweep():
    """
    Training function for sweep
    Called by W&B sweep agent
    """
    # Initialize run
    run = wandb.init()
    
    # Get config from sweep
    config = wandb.config
    
    # Train model (using previous function)
    # model = train_with_wandb(X, y, config)
    
    # W&B automatically logs metrics


# Initialize sweep
# sweep_id = wandb.sweep (sweep_config, project="trading-strategy")

# Run sweep agent
# wandb.agent (sweep_id, function=train_sweep, count=20)  # 20 runs

print("Sweep configuration defined")
print("To run: wandb agent <sweep_id>")
\`\`\`

---

## Experiment Comparison and Analysis

### Comparing Multiple Experiments

\`\`\`python
"""
Experiment Comparison Tools
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentComparer:
    """
    Compare experiments systematically
    """
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.client = mlflow.tracking.MlflowClient()
        
        # Get experiment
        self.experiment = self.client.get_experiment_by_name (experiment_name)
        
        if self.experiment is None:
            raise ValueError (f"Experiment '{experiment_name}' not found")
    
    def get_all_runs (self, max_results=100):
        """
        Get all runs from experiment
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            max_results=max_results
        )
        
        return runs
    
    def runs_to_dataframe (self):
        """
        Convert runs to DataFrame for analysis
        """
        runs = self.get_all_runs()
        
        data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
                'start_time': pd.to_datetime (run.info.start_time, unit='ms'),
                'duration_seconds': (run.info.end_time - run.info.start_time) / 1000
                if run.info.end_time else None
            }
            
            # Add parameters
            run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
            
            # Add metrics
            run_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            
            data.append (run_data)
        
        df = pd.DataFrame (data)
        
        return df
    
    def plot_metrics_comparison (self, metric_name='val_rmse'):
        """
        Plot metric across runs
        """
        df = self.runs_to_dataframe()
        
        metric_col = f"metric_{metric_name}"
        
        if metric_col not in df.columns:
            print(f"Metric '{metric_name}' not found")
            return
        
        # Sort by metric
        df = df.sort_values (metric_col)
        
        # Plot
        fig, ax = plt.subplots (figsize=(12, 6))
        
        ax.bar (range (len (df)), df[metric_col])
        ax.set_xlabel('Run (sorted by performance)')
        ax.set_ylabel (metric_name)
        ax.set_title (f'{metric_name} Across All Runs')
        
        # Highlight best
        best_idx = df[metric_col].idxmin()
        ax.bar (best_idx, df.loc[best_idx, metric_col], color='green', label='Best')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        # Print top 5
        print(f"\\nTop 5 runs by {metric_name}:")
        print(df[['run_name', metric_col]].head(5))
    
    def plot_param_vs_metric (self, param_name, metric_name='val_rmse'):
        """
        Plot relationship between parameter and metric
        """
        df = self.runs_to_dataframe()
        
        param_col = f"param_{param_name}"
        metric_col = f"metric_{metric_name}"
        
        if param_col not in df.columns or metric_col not in df.columns:
            print(f"Column not found: {param_col} or {metric_col}")
            return
        
        # Convert param to numeric if possible
        try:
            df[param_col] = pd.to_numeric (df[param_col])
        except:
            pass
        
        # Plot
        fig, ax = plt.subplots (figsize=(10, 6))
        
        ax.scatter (df[param_col], df[metric_col], alpha=0.6, s=100)
        ax.set_xlabel (param_name)
        ax.set_ylabel (metric_name)
        ax.set_title (f'{param_name} vs {metric_name}')
        
        # Trend line if numeric
        if pd.api.types.is_numeric_dtype (df[param_col]):
            z = np.polyfit (df[param_col], df[metric_col], 1)
            p = np.poly1d (z)
            ax.plot (df[param_col], p (df[param_col]), "r--", alpha=0.8, label='Trend')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_best_run (self, metric='val_rmse', mode='min'):
        """
        Get best run by metric
        """
        df = self.runs_to_dataframe()
        
        metric_col = f"metric_{metric}"
        
        if metric_col not in df.columns:
            print(f"Metric '{metric}' not found")
            return None
        
        if mode == 'min':
            best_idx = df[metric_col].idxmin()
        else:
            best_idx = df[metric_col].idxmax()
        
        best_run = df.loc[best_idx]
        
        print(f"\\n🏆 Best run:")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  {metric}: {best_run[metric_col]:.6f}")
        
        # Print all parameters
        param_cols = [col for col in df.columns if col.startswith('param_')]
        print(f"\\n  Parameters:")
        for col in param_cols:
            param_name = col.replace('param_', ')
            print(f"    {param_name}: {best_run[col]}")
        
        return best_run


# Example usage (uncomment if you have MLflow runs)
# comparer = ExperimentComparer("trading_strategy_development")
# df = comparer.runs_to_dataframe()
# print(f"\\nTotal runs: {len (df)}")
# comparer.plot_metrics_comparison('val_rmse')
# comparer.plot_param_vs_metric('max_depth', 'val_rmse')
# best = comparer.get_best_run('val_rmse', 'min')
\`\`\`

---

## Data and Model Versioning

### DVC for Data Versioning

\`\`\`python
"""
Data Version Control (DVC)
Simplified version - actual DVC uses Git
"""

import hashlib
import json
from pathlib import Path
import shutil

class SimpleDataVersionControl:
    """
    Simplified DVC-like system
    
    Real DVC:
    - Integrates with Git
    - Stores data in S3/GCS
    - Tracks with .dvc files
    """
    
    def __init__(self, storage_dir=".dvc_storage"):
        self.storage_dir = Path (storage_dir)
        self.storage_dir.mkdir (exist_ok=True)
        
        self.metadata_file = self.storage_dir / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions (self):
        """Load version metadata"""
        if self.metadata_file.exists():
            with open (self.metadata_file, 'r') as f:
                return json.load (f)
        return {}
    
    def _save_versions (self):
        """Save version metadata"""
        with open (self.metadata_file, 'w') as f:
            json.dump (self.versions, f, indent=2)
    
    def _compute_hash (self, file_path):
        """Compute file hash"""
        import hashlib
        
        hash_md5 = hashlib.md5()
        with open (file_path, "rb") as f:
            for chunk in iter (lambda: f.read(4096), b""):
                hash_md5.update (chunk)
        
        return hash_md5.hexdigest()
    
    def add (self, file_path, tag=None):
        """
        Add file to version control
        """
        file_path = Path (file_path)
        
        if not file_path.exists():
            raise FileNotFoundError (f"File not found: {file_path}")
        
        # Compute hash
        file_hash = self._compute_hash (file_path)
        
        # Store file
        storage_path = self.storage_dir / file_hash
        shutil.copy (file_path, storage_path)
        
        # Save metadata
        version_info = {
            "hash": file_hash,
            "original_path": str (file_path),
            "tag": tag,
            "timestamp": pd.Timestamp.now().isoformat(),
            "size_bytes": file_path.stat().st_size
        }
        
        self.versions[file_hash] = version_info
        
        if tag:
            self.versions[f"tag:{tag}"] = file_hash
        
        self._save_versions()
        
        print(f"✓ Added {file_path.name}")
        print(f"  Hash: {file_hash[:8]}...")
        if tag:
            print(f"  Tag: {tag}")
        
        return file_hash
    
    def get (self, identifier):
        """
        Get file by hash or tag
        """
        # Check if tag
        if not identifier.startswith("tag:") and f"tag:{identifier}" in self.versions:
            identifier = self.versions[f"tag:{identifier}"]
        
        if identifier not in self.versions:
            raise ValueError (f"Version not found: {identifier}")
        
        version_info = self.versions[identifier]
        storage_path = self.storage_dir / version_info['hash']
        
        return storage_path, version_info
    
    def list_versions (self):
        """List all versions"""
        print("\\n=== Data Versions ===")
        
        for key, value in self.versions.items():
            if key.startswith("tag:"):
                continue
            
            if isinstance (value, dict):
                print(f"\\nHash: {value['hash'][:8]}...")
                print(f"  Path: {value['original_path']}")
                print(f"  Tag: {value.get('tag', 'none')}")
                print(f"  Time: {value['timestamp']}")
                print(f"  Size: {value['size_bytes'] / 1024:.1f} KB")


# Example usage
dvc = SimpleDataVersionControl()

# Create sample file
import pandas as pd
sample_data = pd.DataFrame({
    'price': np.random.randn(1000)
})

sample_data.to_csv('data.csv', index=False)

# Add to version control
file_hash = dvc.add('data.csv', tag='v1.0')

# Modify data
sample_data['volume'] = np.random.randn(1000)
sample_data.to_csv('data.csv', index=False)

# Add new version
file_hash_v2 = dvc.add('data.csv', tag='v1.1')

# List versions
dvc.list_versions()

# Get specific version
storage_path, info = dvc.get('v1.0')
print(f"\\nRetrieved v1.0 from: {storage_path}")
\`\`\`

---

## Reproducibility Best Practices

### Complete Reproducibility Checklist

\`\`\`python
"""
Reproducibility Framework
"""

class ReproducibleExperiment:
    """
    Ensure experiments are fully reproducible
    """
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.config = {}
    
    def setup_environment (self):
        """
        1. Environment and dependencies
        """
        import sys
        import platform
        
        env_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "dependencies": self._get_dependencies()
        }
        
        self.config['environment'] = env_info
        
        print("✓ Environment captured")
        
        return env_info
    
    def _get_dependencies (self):
        """Get installed packages"""
        import pkg_resources
        
        installed_packages = {
            pkg.key: pkg.version
            for pkg in pkg_resources.working_set
        }
        
        return installed_packages
    
    def set_random_seeds (self, seed=42):
        """
        2. Random seeds for reproducibility
        """
        import random
        import numpy as np
        
        random.seed (seed)
        np.random.seed (seed)
        
        # PyTorch
        try:
            import torch
            torch.manual_seed (seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all (seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # TensorFlow
        try:
            import tensorflow as tf
            tf.random.set_seed (seed)
        except ImportError:
            pass
        
        self.config['random_seed'] = seed
        
        print(f"✓ Random seeds set: {seed}")
    
    def version_data (self, data_path):
        """
        3. Data versioning
        """
        import hashlib
        
        # Compute data hash
        with open (data_path, 'rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        
        self.config['data'] = {
            "path": data_path,
            "hash": data_hash,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        print(f"✓ Data versioned: {data_hash[:8]}...")
        
        return data_hash
    
    def version_code (self):
        """
        4. Code versioning (Git commit)
        """
        try:
            import subprocess
            
            # Get git commit
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode('ascii').strip()
            
            # Get branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            ).decode('ascii').strip()
            
            self.config['code'] = {
                "commit": commit_hash,
                "branch": branch
            }
            
            print(f"✓ Code versioned: {commit_hash[:8]}...")
            
            return commit_hash
        
        except Exception as e:
            print(f"⚠️  Could not version code: {e}")
            return None
    
    def save_config (self, path='experiment_config.json'):
        """
        Save complete configuration
        """
        with open (path, 'w') as f:
            json.dump (self.config, f, indent=2, default=str)
        
        print(f"✓ Config saved: {path}")
    
    def run_reproducible_experiment (self, train_fn, **kwargs):
        """
        Run fully reproducible experiment
        """
        print(f"\\n=== Reproducible Experiment: {self.experiment_name} ===\\n")
        
        # 1. Setup
        self.setup_environment()
        self.set_random_seeds()
        
        # 2. Version control
        # self.version_code()
        
        # 3. Run experiment
        result = train_fn(**kwargs)
        
        # 4. Save config
        self.save_config()
        
        print(f"\\n✓ Experiment complete")
        
        return result


# Example usage
def my_train_function(X, y):
    """Example training function"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor (n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    rmse = np.sqrt (mean_squared_error (y_val, val_preds))
    
    print(f"  Val RMSE: {rmse:.6f}")
    
    return model, rmse

# Run reproducible experiment
experiment = ReproducibleExperiment("trading_model_v1")

X = np.random.randn(1000, 20)
y = X[:, 0] * 0.5 + np.random.randn(1000) * 0.1

result = experiment.run_reproducible_experiment(
    my_train_function,
    X=X,
    y=y
)
\`\`\`

---

## Key Takeaways

1. **Track Everything**: Parameters, metrics, code, data, environment
2. **Use Tools**: MLflow for simplicity, W&B for visualizations, Neptune for teams
3. **Model Registry**: Manage model lifecycle (Staging → Production)
4. **Compare Systematically**: Use tables, plots to compare experiments
5. **Version Data**: Use DVC or similar for data versioning
6. **Reproducibility**: Set seeds, version code, track environment
7. **Collaboration**: Shared experiment tracking enables team collaboration

**Trading-Specific Tips**:
- Track trading metrics (Sharpe ratio, max drawdown, win rate)
- Version backtesting data (avoid lookahead bias)
- Compare strategies head-to-head
- Track infrastructure costs per experiment

**Next Steps**: With experiment tracking in place, we'll build robust model training pipelines in the next section.
`,
};
