export const mlModelServingTrading = {
  title: 'ML Model Serving for Trading',
  id: 'ml-model-serving-trading',
  content: `
# ML Model Serving for Trading

## Introduction

Modern trading strategies increasingly rely on **machine learning models** for:
- Price prediction
- Signal generation
- Risk assessment
- Order routing optimization
- Market regime detection

**Challenge**: ML models must deliver predictions in **<1ms** for high-frequency trading, with feature computation, model inference, and result delivery all included.

### Requirements

- **Ultra-low latency**: <1ms end-to-end (feature → prediction)
- **High throughput**: 100K+ predictions/sec
- **Model freshness**: Update models without downtime
- **Feature consistency**: Training features match serving features
- **Monitoring**: Track prediction quality, drift, latency

By the end of this section, you'll understand:
- Sub-millisecond model inference
- Real-time feature computation
- Model deployment strategies
- A/B testing in production
- Monitoring and drift detection

---

## Architecture Overview

\`\`\`
Market Data Stream
    ↓
Feature Store (Redis) ← Feature Engineering Pipeline
    ↓
Model Server (in-memory)
    ↓
Trading Strategy
    ↓
Order Management
\`\`\`

---

## Sub-Millisecond Inference

### Model Selection for Speed

**Fast models** (<100μs inference):
- Linear models (Ridge, Lasso)
- Tree ensembles (LightGBM, XGBoost with depth limit)
- Simple neural networks (1-2 layers)

**Slow models** (>1ms inference):
- Deep neural networks (10+ layers)
- Transformers
- Graph neural networks

### Optimized Inference

\`\`\`python
"""
Optimized ML Model Serving
<1ms inference for trading
"""

import numpy as np
from typing import Dict
import time

class FastModelServer:
    """
    Ultra-low-latency model serving
    Target: <100μs inference
    """
    
    def __init__(self, model_path: str):
        # Load model into memory
        self.model = self.load_model(model_path)
        
        # Pre-allocate arrays (avoid allocation during inference)
        self.feature_buffer = np.zeros(100, dtype=np.float32)
        self.prediction_buffer = np.zeros(1, dtype=np.float32)
        
        # Warm up (first inference slower due to cache misses)
        self.warmup()
    
    def load_model(self, path: str):
        """Load model optimized for inference"""
        # Option 1: LightGBM (very fast)
        import lightgbm as lgb
        model = lgb.Booster(model_file=path)
        return model
        
        # Option 2: ONNX Runtime (optimized)
        # import onnxruntime as ort
        # model = ort.InferenceSession(path)
        # return model
    
    def warmup(self, n_iterations: int = 100):
        """Warm up model (load into CPU cache)"""
        dummy_features = np.random.randn(100).astype(np.float32)
        for _ in range(n_iterations):
            _ = self.predict(dummy_features)
    
    def predict(self, features: np.ndarray) -> float:
        """
        Make prediction
        Target: <100μs
        """
        # Avoid array copy if possible
        # Use pre-allocated buffer
        
        # LightGBM prediction
        pred = self.model.predict([features])[0]
        
        return pred

# Benchmark
model_server = FastModelServer("model.txt")

latencies = []
for _ in range(10000):
    features = np.random.randn(100).astype(np.float32)
    
    start = time.perf_counter()
    pred = model_server.predict(features)
    end = time.perf_counter()
    
    latencies.append((end - start) * 1_000_000)  # microseconds

print(f"Median latency: {np.median(latencies):.1f}μs")
print(f"P95 latency: {np.percentile(latencies, 95):.1f}μs")
print(f"P99 latency: {np.percentile(latencies, 99):.1f}μs")

# Typical results:
# LightGBM (100 trees, depth 6): 20-50μs
# Linear model: 5-10μs
# Simple NN (2 layers, 64 units): 50-100μs
\`\`\`

### ONNX Runtime Optimization

\`\`\`python
"""
ONNX Runtime for optimized inference
Works with PyTorch, TensorFlow, scikit-learn models
"""

import onnx
import onnxruntime as ort
import numpy as np

class ONNXModelServer:
    """
    ONNX Runtime model server
    Optimized for CPU inference
    """
    
    def __init__(self, onnx_path: str):
        # Create inference session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Single thread (avoid overhead)
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Fast prediction"""
        # Ensure correct shape [batch_size, features]
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Run inference
        pred = self.session.run(
            [self.output_name],
            {self.input_name: features.astype(np.float32)}
        )[0]
        
        return pred[0]

# Convert scikit-learn model to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onx = convert_sklearn(model, initial_types=initial_type)

# Save
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Serve with ONNX Runtime
server = ONNXModelServer("model.onnx")

# ONNX Runtime typically 2-5x faster than scikit-learn
\`\`\`

---

## Real-Time Feature Engineering

### Feature Store Architecture

\`\`\`python
"""
Real-time feature store using Redis
Compute features once, serve to multiple models
"""

import redis
import json
from typing import Dict, List
import numpy as np

class FeatureStore:
    """
    Real-time feature store
    Store pre-computed features in Redis
    """
    
    def __init__(self, redis_host: str = 'localhost'):
        self.redis = redis.Redis(
            host=redis_host,
            decode_responses=False,  # Binary for speed
            socket_keepalive=True,
            socket_connect_timeout=1
        )
        
        # Feature TTL (time-to-live)
        self.feature_ttl = 60  # 60 seconds
    
    def compute_features(self, symbol: str, market_data: Dict) -> np.ndarray:
        """
        Compute features from market data
        """
        features = []
        
        # Price-based features
        features.append(market_data['close'] / market_data['open'] - 1)  # Intrabar return
        features.append(market_data['high'] / market_data['low'] - 1)  # Range
        features.append(market_data['close'])  # Price level
        
        # Volume
        features.append(market_data['volume'])
        features.append(np.log(market_data['volume'] + 1))  # Log volume
        
        # Technical indicators (pre-computed)
        historical_features = self.get_historical_features(symbol)
        if historical_features:
            features.extend(historical_features)
        
        return np.array(features, dtype=np.float32)
    
    def store_features(self, symbol: str, features: np.ndarray):
        """Store features in Redis"""
        key = f"features:{symbol}"
        
        # Serialize to bytes (faster than JSON)
        features_bytes = features.tobytes()
        
        # Store with TTL
        self.redis.setex(key, self.feature_ttl, features_bytes)
    
    def get_features(self, symbol: str) -> Optional[np.ndarray]:
        """Retrieve features from Redis"""
        key = f"features:{symbol}"
        
        features_bytes = self.redis.get(key)
        if not features_bytes:
            return None
        
        # Deserialize
        features = np.frombuffer(features_bytes, dtype=np.float32)
        
        return features
    
    def get_historical_features(self, symbol: str) -> List[float]:
        """
        Get pre-computed historical features
        (SMA, RSI, etc computed by batch pipeline)
        """
        key = f"historical:{symbol}"
        data = self.redis.get(key)
        
        if not data:
            return []
        
        return json.loads(data)

# Feature pipeline (runs continuously)
class FeaturePipeline:
    """
    Compute and update features in real-time
    """
    
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
    
    def on_market_data(self, symbol: str, market_data: Dict):
        """
        Process market data tick
        Compute and store features
        """
        # Compute features
        features = self.store.compute_features(symbol, market_data)
        
        # Store in Redis
        self.store.store_features(symbol, features)

# Usage
store = FeatureStore()
pipeline = FeaturePipeline(store)

# On market data tick
pipeline.on_market_data('AAPL', {
    'open': 150.0,
    'high': 151.0,
    'low': 149.5,
    'close': 150.5,
    'volume': 1000000
})

# Model serving retrieves features
features = store.get_features('AAPL')
if features is not None:
    prediction = model.predict(features)
\`\`\`

---

## Model Deployment Strategies

### Blue-Green Deployment

\`\`\`python
"""
Blue-Green deployment for zero-downtime model updates
"""

class BlueGreenModelServer:
    """
    Run two model versions simultaneously
    Switch traffic instantly
    """
    
    def __init__(self):
        self.blue_model = None  # Current production model
        self.green_model = None  # New model (being tested)
        self.active_model = 'blue'  # Which model is serving traffic
    
    def deploy_green(self, model_path: str):
        """Deploy new model to green"""
        print("Loading new model to green...")
        self.green_model = FastModelServer(model_path)
        print("Green model ready")
    
    def switch_to_green(self):
        """Switch traffic from blue to green"""
        if self.green_model is None:
            raise Exception("Green model not loaded")
        
        print("Switching traffic to green...")
        self.active_model = 'green'
        
        # Old blue model can be unloaded after monitoring period
    
    def switch_to_blue(self):
        """Rollback to blue if green has issues"""
        print("Rolling back to blue...")
        self.active_model = 'blue'
    
    def predict(self, features: np.ndarray) -> float:
        """Route prediction to active model"""
        if self.active_model == 'blue':
            return self.blue_model.predict(features)
        else:
            return self.green_model.predict(features)

# Deployment process
server = BlueGreenModelServer()

# 1. Blue model serving production
server.blue_model = FastModelServer("model_v1.txt")
server.active_model = 'blue'

# 2. Deploy new model to green
server.deploy_green("model_v2.txt")

# 3. Test green model (shadow mode)
# ... validate predictions ...

# 4. Switch to green
server.switch_to_green()

# 5. Monitor for 1 hour

# 6. If issues, rollback
# server.switch_to_blue()
\`\`\`

### Canary Deployment

\`\`\`python
"""
Canary deployment: Gradually shift traffic to new model
"""

import random

class CanaryModelServer:
    """
    Route percentage of traffic to new model
    Gradually increase if performance good
    """
    
    def __init__(self):
        self.old_model = None
        self.new_model = None
        self.canary_percentage = 0  # Start with 0% on new model
    
    def set_canary_percentage(self, percentage: int):
        """Set percentage of traffic to new model"""
        self.canary_percentage = percentage
        print(f"Canary set to {percentage}%")
    
    def predict(self, features: np.ndarray) -> tuple[float, str]:
        """
        Route to old or new model based on canary percentage
        Returns: (prediction, model_used)
        """
        if random.random() * 100 < self.canary_percentage:
            # Route to new model
            pred = self.new_model.predict(features)
            return pred, 'new'
        else:
            # Route to old model
            pred = self.old_model.predict(features)
            return pred, 'old'

# Canary rollout
server = CanaryModelServer()
server.old_model = FastModelServer("model_v1.txt")
server.new_model = FastModelServer("model_v2.txt")

# Day 1: 5% traffic to new model
server.set_canary_percentage(5)

# Monitor for 24 hours
# If performance good:

# Day 2: 25% traffic
server.set_canary_percentage(25)

# Day 3: 50%
server.set_canary_percentage(50)

# Day 4: 100% (full rollout)
server.set_canary_percentage(100)
\`\`\`

---

## A/B Testing in Production

\`\`\`python
"""
A/B test trading strategies with ML models
"""

class ABTestFramework:
    """
    A/B test models in production
    Track performance metrics per model
    """
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
    
    def add_model(self, model_id: str, model, traffic_percentage: int):
        """Add model variant"""
        self.models[model_id] = {
            'model': model,
            'percentage': traffic_percentage
        }
        self.metrics[model_id] = {
            'predictions': 0,
            'latency': [],
            'accuracy': [],  # If we can measure
        }
    
    def route_request(self, features: np.ndarray) -> tuple[float, str]:
        """Route to model based on traffic allocation"""
        # Weighted random selection
        rand = random.random() * 100
        cumulative = 0
        
        for model_id, config in self.models.items():
            cumulative += config['percentage']
            if rand < cumulative:
                # Selected this model
                start = time.perf_counter()
                pred = config['model'].predict(features)
                latency = (time.perf_counter() - start) * 1_000_000  # μs
                
                # Record metrics
                self.metrics[model_id]['predictions'] += 1
                self.metrics[model_id]['latency'].append(latency)
                
                return pred, model_id
        
        # Fallback to first model
        model_id = list(self.models.keys())[0]
        pred = self.models[model_id]['model'].predict(features)
        return pred, model_id
    
    def get_metrics(self) -> Dict:
        """Get A/B test metrics"""
        summary = {}
        for model_id, metrics in self.metrics.items():
            summary[model_id] = {
                'predictions': metrics['predictions'],
                'median_latency_us': np.median(metrics['latency']),
                'p95_latency_us': np.percentile(metrics['latency'], 95),
            }
        return summary

# A/B test setup
ab_test = ABTestFramework()

# Model A: Current production (70% traffic)
ab_test.add_model('model_a', FastModelServer("model_a.txt"), 70)

# Model B: New experimental (30% traffic)
ab_test.add_model('model_b', FastModelServer("model_b.txt"), 30)

# Run for 1 week
for _ in range(100000):
    features = np.random.randn(100).astype(np.float32)
    pred, model_used = ab_test.route_request(features)

# Analyze results
metrics = ab_test.get_metrics()
print(json.dumps(metrics, indent=2))

# Decision: If model_b has better Sharpe ratio and acceptable latency → full rollout
\`\`\`

---

## Monitoring and Drift Detection

\`\`\`python
"""
Monitor model performance and detect drift
"""

class ModelMonitor:
    """
    Monitor ML models in production
    """
    
    def __init__(self):
        self.predictions = []
        self.features = []
        self.actuals = []  # If we can observe outcomes
    
    def log_prediction(self, features: np.ndarray, prediction: float):
        """Log prediction for monitoring"""
        self.features.append(features)
        self.predictions.append(prediction)
    
    def log_actual(self, actual: float):
        """Log actual outcome (if observable)"""
        self.actuals.append(actual)
    
    def detect_feature_drift(self) -> Dict:
        """
        Detect if feature distribution has changed
        Compare recent features to training distribution
        """
        recent_features = np.array(self.features[-1000:])  # Last 1000 predictions
        
        drift_scores = {}
        for i in range(recent_features.shape[1]):
            # Compare mean (simple drift detection)
            recent_mean = recent_features[:, i].mean()
            training_mean = self.training_stats['means'][i]
            
            # Deviation in standard deviations
            drift = abs(recent_mean - training_mean) / self.training_stats['stds'][i]
            
            if drift > 3:  # 3 standard deviations
                drift_scores[f'feature_{i}'] = drift
        
        return drift_scores
    
    def calculate_prediction_drift(self) -> float:
        """
        Detect if prediction distribution has changed
        """
        recent_preds = self.predictions[-1000:]
        old_preds = self.predictions[-10000:-1000]
        
        # KL divergence or simpler: mean/std comparison
        recent_mean = np.mean(recent_preds)
        old_mean = np.mean(old_preds)
        
        drift = abs(recent_mean - old_mean)
        
        return drift
    
    def alert_if_drift(self):
        """Check for drift and alert"""
        feature_drift = self.detect_feature_drift()
        prediction_drift = self.calculate_prediction_drift()
        
        if feature_drift:
            print(f"ALERT: Feature drift detected: {feature_drift}")
        
        if prediction_drift > 0.1:
            print(f"ALERT: Prediction drift: {prediction_drift}")

# Run monitoring continuously
monitor = ModelMonitor()

# Log every prediction
for _ in range(10000):
    features = generate_features()
    pred = model.predict(features)
    monitor.log_prediction(features, pred)

# Check for drift hourly
monitor.alert_if_drift()
\`\`\`

---

## Best Practices

### 1. Model Complexity vs Latency

**Target**: <100μs inference
- Linear models: 5-10μs ✓
- LightGBM (100 trees, depth 6): 20-50μs ✓
- Simple NN (2 layers): 50-100μs ✓
- Deep NN (10 layers): 1-10ms ✗

### 2. Feature Computation

**Pre-compute** expensive features:
- Moving averages, RSI, MACD → compute in batch, store in Redis
- Raw features → compute on-the-fly (fast)

### 3. Model Updates

**Frequency**: Retrain daily/weekly, not every tick
**Deployment**: Blue-green or canary (zero downtime)
**Validation**: Shadow mode before production

### 4. Monitoring

**Metrics to track**:
- Latency (p50, p95, p99)
- Throughput (predictions/sec)
- Prediction distribution
- Feature drift
- Model accuracy (if outcomes observable)

---

## Summary

ML model serving for trading requires:

1. **Sub-millisecond inference**: Use optimized models (LightGBM, ONNX), <100μs target
2. **Real-time features**: Redis feature store, pre-compute expensive features
3. **Zero-downtime deployment**: Blue-green or canary rollout
4. **A/B testing**: Compare model variants in production
5. **Monitoring**: Track latency, drift, performance

**Trade-off**: Model complexity vs latency. Simple models fast, complex models accurate. Find optimal balance for your strategy.

This completes Module 20! You now understand how to design production trading systems from order management through ML model serving.
`,
};
