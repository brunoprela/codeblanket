export const deepLearningForTimeSeries = {
  title: 'Deep Learning for Time Series',
  id: 'deep-learning-for-time-series',
  content: `
# Deep Learning for Time Series

## Introduction

While classical models (ARIMA, GARCH) excel with limited data, deep learning shines when you have:
- **Large datasets** (10,000+ observations)
- **Non-linear patterns** (regime changes, complex interactions)
- **Multiple related series** (portfolio of 100+ stocks)
- **High-dimensional features** (technical indicators, alternative data)

This section covers:
- LSTMs for sequence modeling
- 1D CNNs for pattern recognition
- Temporal Convolutional Networks (TCN)
- Transformers for time series
- Attention mechanisms
- Practical trading applications

### Why Deep Learning for Finance?

**Advantages**:
- Captures non-linear relationships (momentum × volatility interactions)
- Handles high-dimensional inputs (100+ features)
- Learns hierarchical patterns (local + global)
- No feature engineering required (learns representations)

**Disadvantages**:
- Requires large datasets (classical models win with < 1000 samples)
- Black box (harder to interpret)
- Computationally expensive (GPU needed for training)
- Risk of overfitting (millions of parameters)

---

## LSTM Networks

### Long Short-Term Memory

LSTMs solve the **vanishing gradient problem** in RNNs, enabling long-term dependencies.

**LSTM Cell Structure**:
\`\`\`
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell update:  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell state:   C_t = f_t * C_{t-1} + i_t * C̃_t
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t * tanh(C_t)
\`\`\`

### Implementing LSTM for Stock Prediction

\`\`\`python
"""
LSTM for Stock Price Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
spy = yf.download('SPY', start='2015-01-01', end='2024-01-01')
prices = spy['Close'].values

print(f"Dataset size: {len (prices)} days")

# Create sequences
def create_sequences (data, seq_length=60, pred_horizon=1):
    """
    Create sequences for supervised learning
    
    Args:
        data: Time series array
        seq_length: Look-back window
        pred_horizon: Days ahead to predict
    
    Returns:
        X: Input sequences [samples, seq_length]
        y: Target values [samples]
    """
    X, y = [], []
    
    for i in range (len (data) - seq_length - pred_horizon + 1):
        X.append (data[i:i+seq_length])
        y.append (data[i+seq_length+pred_horizon-1])
    
    return np.array(X), np.array (y)

# Create sequences
seq_length = 60  # 60 days look-back
X, y = create_sequences (prices, seq_length=seq_length, pred_horizon=1)

print(f"Sequences: {X.shape}, Targets: {y.shape}")

# Train/test split (temporal)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Normalize (fit on train only!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform (y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform (y_test.reshape(-1, 1)).flatten()

print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor (y)
    
    def __len__(self):
        return len (self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader (train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader (test_dataset, batch_size=32, shuffle=False)

# LSTM Model
class LSTMPredictor (nn.Module):
    """
    LSTM model for price prediction
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear (hidden_size, 1)
    
    def forward (self, x):
        # x shape: [batch, seq_length]
        x = x.unsqueeze(-1)  # [batch, seq_length, 1]
        
        # LSTM forward
        lstm_out, _ = self.lstm (x)  # [batch, seq_length, hidden_size]
        
        # Use last time step
        last_output = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # Prediction
        pred = self.fc (last_output)  # [batch, 1]
        
        return pred.squeeze()

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPredictor (input_size=1, hidden_size=50, num_layers=2, dropout=0.2).to (device)

print(f"\\nModel: {model}")
print(f"Parameters: {sum (p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam (model.parameters(), lr=0.001)

def train_epoch (model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to (device)
        y_batch = y_batch.to (device)
        
        # Forward
        pred = model(X_batch)
        loss = criterion (pred, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len (loader)

def evaluate (model, loader, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to (device)
            y_batch = y_batch.to (device)
            
            pred = model(X_batch)
            loss = criterion (pred, y_batch)
            
            total_loss += loss.item()
            predictions.extend (pred.cpu().numpy())
            actuals.extend (y_batch.cpu().numpy())
    
    return total_loss / len (loader), np.array (predictions), np.array (actuals)

# Training loop
epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

train_losses = []
val_losses = []

print("\\n=== Training LSTM ===")
for epoch in range (epochs):
    train_loss = train_epoch (model, train_loader, criterion, optimizer)
    val_loss, _, _ = evaluate (model, test_loader, criterion)
    
    train_losses.append (train_loss)
    val_losses.append (val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save (model.state_dict(), 'best_lstm.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict (torch.load('best_lstm.pth'))

# Evaluate
_, test_preds, test_actuals = evaluate (model, test_loader, criterion)

# Inverse transform predictions
test_preds_original = scaler_y.inverse_transform (test_preds.reshape(-1, 1)).flatten()
test_actuals_original = scaler_y.inverse_transform (test_actuals.reshape(-1, 1)).flatten()

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error (test_actuals_original, test_preds_original)
rmse = np.sqrt (mean_squared_error (test_actuals_original, test_preds_original))
mape = np.mean (np.abs((test_actuals_original - test_preds_original) / test_actuals_original)) * 100
r2 = r2_score (test_actuals_original, test_preds_original)

print("\\n=== Test Results ===")
print(f"MAE: \${mae:.2f}")
print(f"RMSE: \${rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")

# Plot predictions
plt.figure (figsize = (14, 6))
plt.plot (test_actuals_original, label = 'Actual', alpha = 0.7)
plt.plot (test_preds_original, label = 'LSTM Prediction', alpha = 0.7)
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('LSTM Price Predictions')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()

# Plot training curves
plt.figure (figsize = (10, 5))
plt.plot (train_losses, label = 'Train Loss')
plt.plot (val_losses, label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
\`\`\`

---

## 1D Convolutional Neural Networks

### Pattern Recognition in Time Series

1D CNNs detect **local patterns** (like technical chart patterns) efficiently.

\`\`\`python
"""
1D CNN for Time Series
"""

class CNN1DPredictor (nn.Module):
    """
    1D CNN for time series prediction
    """
    
    def __init__(self, seq_length=60):
        super(CNN1DPredictor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv layer 1
            nn.Conv1d (in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d (kernel_size=2),
            nn.Dropout(0.2),
            
            # Conv layer 2
            nn.Conv1d (in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d (kernel_size=2),
            nn.Dropout(0.2),
            
            # Conv layer 3
            nn.Conv1d (in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d (kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # Calculate flattened size
        conv_output_size = seq_length // (2 * 2 * 2)  # After 3 pooling layers
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * conv_output_size, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 1)
        )
    
    def forward (self, x):
        # x shape: [batch, seq_length]
        x = x.unsqueeze(1)  # [batch, 1, seq_length]
        
        # Convolutional layers
        x = self.conv_layers (x)  # [batch, 128, reduced_length]
        
        # Flatten
        x = x.view (x.size(0), -1)
        
        # Fully connected
        pred = self.fc_layers (x)
        
        return pred.squeeze()

# Train CNN model
cnn_model = CNN1DPredictor (seq_length=seq_length).to (device)

print(f"\\nCNN Model: {cnn_model}")
print(f"Parameters: {sum (p.numel() for p in cnn_model.parameters()):,}")

# Training (same loop as LSTM)
optimizer_cnn = torch.optim.Adam (cnn_model.parameters(), lr=0.001)

print("\\n=== Training 1D CNN ===")
for epoch in range(30):
    train_loss = train_epoch (cnn_model, train_loader, criterion, optimizer_cnn)
    val_loss, _, _ = evaluate (cnn_model, test_loader, criterion)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Evaluate CNN
_, cnn_preds, _ = evaluate (cnn_model, test_loader, criterion)
cnn_preds_original = scaler_y.inverse_transform (cnn_preds.reshape(-1, 1)).flatten()

cnn_mae = mean_absolute_error (test_actuals_original, cnn_preds_original)
cnn_rmse = np.sqrt (mean_squared_error (test_actuals_original, cnn_preds_original))
cnn_r2 = r2_score (test_actuals_original, cnn_preds_original)

print("\\n=== CNN Test Results ===")
print(f"MAE: \${cnn_mae:.2f}")
print(f"RMSE: \${cnn_rmse:.2f}")
print(f"R²: {cnn_r2:.4f}")
\`\`\`

---

## Temporal Convolutional Networks (TCN)

### Dilated Causal Convolutions

TCNs use **dilated convolutions** for long-range dependencies without recurrence.

\`\`\`python
"""
Temporal Convolutional Network
"""

class TemporalBlock (nn.Module):
    """
    TCN building block with dilated causal convolutions
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0)  # Causal: remove future
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout (dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout (dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d (in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward (self, x):
        out = self.net (x)
        res = x if self.downsample is None else self.downsample (x)
        return self.relu (out + res)

class TCN(nn.Module):
    """
    Temporal Convolutional Network
    """
    
    def __init__(self, num_channels=[32, 64, 128], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len (num_channels)
        
        for i in range (num_levels):
            dilation = 2 ** i
            in_ch = 1 if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation, dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear (num_channels[-1], 1)
    
    def forward (self, x):
        # x: [batch, seq_length]
        x = x.unsqueeze(1)  # [batch, 1, seq_length]
        
        # TCN forward
        y = self.network (x)  # [batch, channels, seq_length]
        
        # Use last time step
        y = y[:, :, -1]  # [batch, channels]
        
        # Prediction
        pred = self.fc (y)
        
        return pred.squeeze()

# Train TCN
tcn_model = TCN(num_channels=[32, 64, 128], kernel_size=3, dropout=0.2).to (device)

print(f"\\nTCN Model Parameters: {sum (p.numel() for p in tcn_model.parameters()):,}")

optimizer_tcn = torch.optim.Adam (tcn_model.parameters(), lr=0.001)

print("\\n=== Training TCN ===")
for epoch in range(30):
    train_loss = train_epoch (tcn_model, train_loader, criterion, optimizer_tcn)
    val_loss, _, _ = evaluate (tcn_model, test_loader, criterion)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
\`\`\`

---

## Attention Mechanisms & Transformers

### Multi-Head Attention for Time Series

\`\`\`python
"""
Transformer for Time Series
"""

class TimeSeriesTransformer (nn.Module):
    """
    Transformer encoder for time series prediction
    """
    
    def __init__(self, seq_length=60, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter (torch.randn(1, seq_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder (encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear (d_model, 1)
    
    def forward (self, x):
        # x: [batch, seq_length]
        x = x.unsqueeze(-1)  # [batch, seq_length, 1]
        
        # Project to d_model dimensions
        x = self.input_proj (x)  # [batch, seq_length, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer encoding
        x = self.transformer_encoder (x)  # [batch, seq_length, d_model]
        
        # Use last time step
        x = x[:, -1, :]  # [batch, d_model]
        
        # Prediction
        pred = self.fc (x)
        
        return pred.squeeze()

# Train Transformer
transformer_model = TimeSeriesTransformer(
    seq_length=seq_length,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1
).to (device)

print(f"\\nTransformer Model Parameters: {sum (p.numel() for p in transformer_model.parameters()):,}")

optimizer_transformer = torch.optim.Adam (transformer_model.parameters(), lr=0.0001)

print("\\n=== Training Transformer ===")
for epoch in range(30):
    train_loss = train_epoch (transformer_model, train_loader, criterion, optimizer_transformer)
    val_loss, _, _ = evaluate (transformer_model, test_loader, criterion)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/30 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Evaluate all models
_, transformer_preds, _ = evaluate (transformer_model, test_loader, criterion)
transformer_preds_original = scaler_y.inverse_transform (transformer_preds.reshape(-1, 1)).flatten()

print("\\n=== Model Comparison ===")
print(f"{'Model':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
print("-" * 45)
print(f"{'LSTM':<15} \${mae:>9.2f} \${rmse:>9.2f} {r2:>8.4f}")
print(f"{'1D CNN':<15} \${cnn_mae:>9.2f} \${cnn_rmse:>9.2f} {cnn_r2:>8.4f}")

trans_mae = mean_absolute_error (test_actuals_original, transformer_preds_original)
trans_rmse = np.sqrt (mean_squared_error (test_actuals_original, transformer_preds_original))
trans_r2 = r2_score (test_actuals_original, transformer_preds_original)

print(f"{'Transformer':<15} \${trans_mae:>9.2f} \${trans_rmse:>9.2f} {trans_r2:>8.4f}")
\`\`\`

---

## Multi-Horizon Forecasting

### Predicting Multiple Steps Ahead

\`\`\`python
"""
Multi-Horizon LSTM
"""

class MultiHorizonLSTM(nn.Module):
    """
    LSTM that predicts multiple future time steps
    """
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, horizon=5, dropout=0.2):
        super(MultiHorizonLSTM, self).__init__()
        
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear (hidden_size, horizon)
    
    def forward (self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm (x)
        last_output = lstm_out[:, -1, :]
        pred = self.fc (last_output)  # [batch, horizon]
        return pred

# Create multi-horizon targets
def create_multihorizon_sequences (data, seq_length=60, horizon=5):
    X, y = [], []
    
    for i in range (len (data) - seq_length - horizon + 1):
        X.append (data[i:i+seq_length])
        y.append (data[i+seq_length:i+seq_length+horizon])
    
    return np.array(X), np.array (y)

# Example: Predict next 5 days
X_mh, y_mh = create_multihorizon_sequences (prices, seq_length=60, horizon=5)

print(f"Multi-horizon data: X={X_mh.shape}, y={y_mh.shape}")
\`\`\`

---

## Ensemble Methods

### Combining Multiple Models

\`\`\`python
"""
Ensemble of Deep Learning Models
"""

class EnsemblePredictor:
    """
    Ensemble multiple models with weighted averaging
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len (models)] * len (models)
    
    def predict (self, X, scaler_y):
        """Make ensemble prediction"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to (device)
                pred = model(X_tensor).cpu().numpy()
                predictions.append (pred)
        
        # Weighted average
        ensemble_pred = sum (w * p for w, p in zip (self.weights, predictions))
        
        # Inverse transform
        ensemble_pred_original = scaler_y.inverse_transform (ensemble_pred.reshape(-1, 1)).flatten()
        
        return ensemble_pred_original

# Create ensemble
ensemble = EnsemblePredictor(
    models=[model, cnn_model, transformer_model],
    weights=[0.4, 0.3, 0.3]  # LSTM gets higher weight
)

# Predict
ensemble_preds = ensemble.predict(X_test_scaled, scaler_y)

ensemble_mae = mean_absolute_error (test_actuals_original, ensemble_preds)
ensemble_rmse = np.sqrt (mean_squared_error (test_actuals_original, ensemble_preds))
ensemble_r2 = r2_score (test_actuals_original, ensemble_preds)

print("\\n=== Ensemble Results ===")
print(f"MAE: \${ensemble_mae:.2f}")
print(f"RMSE: \${ensemble_rmse:.2f}")
print(f"R²: {ensemble_r2:.4f}")

# Plot all predictions
plt.figure (figsize = (14, 6))
plt.plot (test_actuals_original[: 100], label = 'Actual', color = 'black', linewidth = 2)
plt.plot (test_preds_original[: 100], label = 'LSTM', alpha = 0.7)
plt.plot (cnn_preds_original[: 100], label = 'CNN', alpha = 0.7)
plt.plot (transformer_preds_original[: 100], label = 'Transformer', alpha = 0.7)
plt.plot (ensemble_preds[: 100], label = 'Ensemble', linewidth = 2, linestyle = '--')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Model Comparison (First 100 Test Days)')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()
\`\`\`

---

## Key Takeaways

1. **LSTMs**:
   - Handle long-term dependencies
   - Good for: Sequences with memory requirements
   - Parameters: ~50K-500K (manageable)
   - Training time: Minutes to hours

2. **1D CNNs**:
   - Detect local patterns (chart patterns)
   - Good for: Technical pattern recognition
   - Faster training than LSTM
   - Less memory of long-term dependencies

3. **TCN**:
   - Dilated convolutions for long receptive field
   - Parallel training (faster than LSTM)
   - Good for: Long sequences with distant dependencies
   - More stable training than LSTM

4. **Transformers**:
   - Attention mechanism captures global dependencies
   - Good for: Complex patterns, multi-horizon forecasting
   - High parameter count (100K+)
   - Requires more data than LSTM

5. **Ensemble**:
   - Combine strengths of multiple models
   - Reduces overfitting
   - Improves robustness
   - Typical improvement: 5-15% over best single model

6. **Practical Tips**:
   - Start simple (LSTM) before complex (Transformer)
   - Use early stopping (patience=10-20)
   - Normalize inputs (StandardScaler)
   - Use walk-forward validation
   - Monitor overfitting (train vs validation loss)

7. **When to Use**:
   - **< 1K samples**: Classical models (ARIMA, GARCH)
   - **1K-10K samples**: LSTM, simple CNN
   - **10K+ samples**: Transformers, complex ensembles
   - **Multiple assets**: Multi-task learning

**Next**: Practical trading with real financial data sources and APIs.
`,
};
