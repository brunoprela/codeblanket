export const realTimeMlSystems = {
  title: 'Real-Time ML Systems',
  id: 'real-time-ml-systems',
  content: `
# Real-Time ML Systems

## Introduction

**"Batch processing is easy. Real-time is where the fun begins."**

Real-time ML systems must:
- Process data streams continuously
- Make predictions with <100ms latency
- Update models without downtime
- Handle concept drift in real-time
- Scale to millions of requests per second

**Use Cases**:
- Fraud detection (approve/decline in real-time)
- Trading systems (ms matter)
- Recommendation systems (personalized instantly)
- Anomaly detection (alert immediately)
- Real-time bidding (ad auctions in <100ms)

This section covers building production real-time ML systems.

### Real-Time vs Batch

\`\`\`
Batch ML:
  Offline Training → Batch Predictions → Store Results → Serve
  Latency: Hours to days
  Use: When immediate results not needed

Real-Time ML:
  Stream Processing → Online Predictions → Immediate Response
  Latency: <100ms
  Use: When decisions needed immediately
\`\`\`

---

## Streaming Data Pipelines

### Kafka for ML

\`\`\`python
"""
Real-Time Data Pipeline with Kafka
"""

class KafkaMLPipeline:
    """
    Stream processing pipeline for ML
    
    Architecture:
    1. Kafka: Message broker
    2. Stream processors: Compute features
    3. ML service: Real-time predictions
    4. Output: Predictions to downstream
    """
    
    def __init__(self):
        self.architecture = {
            "data_sources": [
                "Web events",
                "Mobile apps",
                "IoT devices",
                "Transaction systems"
            ],
            "kafka_topics": {
                "raw_events": "Incoming events",
                "features": "Computed features",
                "predictions": "Model outputs",
                "feedback": "User actions (for retraining)"
            },
            "stream_processing": [
                "Apache Flink",
                "Spark Streaming",
                "Kafka Streams"
            ]
        }
    
    def explain_architecture(self):
        """
        Real-time ML architecture
        """
        print("\\n=== Real-Time ML Architecture ===\\n")
        
        print("Flow:")
        print("  1. Data Sources → Kafka (raw_events topic)")
        print("     - Web clicks, transactions, sensor data")
        print("     - High throughput: 100K+ events/sec")
        
        print("\\n  2. Stream Processor → Kafka (features topic)")
        print("     - Compute features in real-time")
        print("     - Join with historical data")
        print("     - Aggregate windows (last 1 hour, 24 hours)")
        
        print("\\n  3. ML Service consumes features")
        print("     - Load model in memory")
        print("     - Predict on each event")
        print("     - P99 latency: <50ms")
        
        print("\\n  4. Predictions → Kafka (predictions topic)")
        print("     - Downstream services consume")
        print("     - Fraud alerts, recommendations, etc.")
        
        print("\\n  5. Feedback loop → Kafka (feedback topic)")
        print("     - User clicks, actual outcomes")
        print("     - Used for model retraining")


# Simulated Kafka consumer/producer
class SimulatedKafkaConsumer:
    """
    Simulated Kafka consumer for ML
    """
    
    def __init__(self, topic):
        self.topic = topic
        self.messages = []
    
    def consume(self, timeout=1.0):
        """
        Consume messages
        """
        import random
        import time
        
        # Simulate receiving a message
        if random.random() > 0.3:  # 70% chance of message
            msg = {
                "user_id": random.randint(1000, 9999),
                "transaction_amount": round(random.uniform(10, 1000), 2),
                "merchant_id": random.randint(100, 999),
                "timestamp": time.time()
            }
            return msg
        
        return None


class RealTimePredictionService:
    """
    Real-time prediction service consuming from Kafka
    """
    
    def __init__(self, model):
        self.model = model
        self.predictions_made = 0
    
    def compute_features(self, event):
        """
        Compute features from event
        
        In production:
        - Lookup historical features from Redis/Feature Store
        - Compute real-time aggregations
        - Join with user profile
        """
        import numpy as np
        
        # Simplified: Just use event data
        features = np.array([[
            event['transaction_amount'],
            event['merchant_id'] % 100,  # Merchant category
            event['timestamp'] % 24,  # Hour of day
            0  # Placeholder for historical features
        ]])
        
        return features
    
    def predict_realtime(self, consumer, producer, max_events=10):
        """
        Real-time prediction loop
        """
        print("\\n=== Real-Time Prediction Service ===\\n")
        print("Listening for events...\\n")
        
        import time
        
        for i in range(max_events):
            # Consume event
            event = consumer.consume()
            
            if event is None:
                time.sleep(0.1)
                continue
            
            # Start timing
            start = time.time()
            
            # Compute features
            features = self.compute_features(event)
            
            # Predict
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]
            
            # Compute latency
            latency_ms = (time.time() - start) * 1000
            
            # Produce prediction
            result = {
                "user_id": event['user_id'],
                "prediction": int(prediction),
                "confidence": float(prediction_proba[prediction]),
                "latency_ms": latency_ms
            }
            
            producer.produce(result)
            
            self.predictions_made += 1
            
            # Log
            label = "FRAUD" if prediction == 1 else "LEGIT"
            print(f"Event {i+1}: User {event['user_id']}, "
                  f"Amount \${event['transaction_amount']: .2f
} → "
                  f"{label} (confidence: {result['confidence']:.2f}, "
                  f"latency: {latency_ms:.1f}ms)")

print(f"\\n✓ Processed {self.predictions_made} events")


class SimulatedKafkaProducer:
"""
    Simulated Kafka producer
"""
    
    def __init__(self, topic):
self.topic = topic
    
    def produce(self, message):
"""
        Produce message to topic
"""
        # In production: kafka.send(topic, message)
pass


# Example: Real - time prediction service
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train model(simplified)
X = np.random.randn(1000, 4)
y = (X[:, 0] > 0.5).astype(int)  # Simple rule
model = RandomForestClassifier(n_estimators = 50, random_state = 42)
model.fit(X, y)

# Setup pipeline
pipeline = KafkaMLPipeline()
pipeline.explain_architecture()

# Setup services
consumer = SimulatedKafkaConsumer('raw_events')
producer = SimulatedKafkaProducer('predictions')
prediction_service = RealTimePredictionService(model)

# Run real - time predictions
prediction_service.predict_realtime(consumer, producer, max_events = 10)
\`\`\`

---

## Low-Latency Inference

### Latency Optimization

\`\`\`python
"""
Optimizing ML Inference for Low Latency
"""

class LatencyOptimization:
    """
    Techniques to achieve <10ms inference
    
    Target: Trading systems, fraud detection, RTB
    """
    
    def __init__(self):
        self.optimization_techniques = {
            "model_optimization": [
                "Model quantization (FP32 → INT8)",
                "Model pruning (remove unimportant weights)",
                "Knowledge distillation (large → small model)",
                "Use simpler models (XGBoost not BERT)"
            ],
            "serving_optimization": [
                "Batch predictions (amortize overhead)",
                "Model caching (keep in memory)",
                "Pre-compute features",
                "Use faster hardware (GPUs for large batches)"
            ],
            "infrastructure": [
                "Co-location (reduce network latency)",
                "Connection pooling",
                "Async processing",
                "Load balancing"
            ]
        }
    
    def model_quantization_example(self):
        """
        Quantization: FP32 → INT8
        
        Benefits:
        - 4x smaller model size
        - 2-4x faster inference
        - Minimal accuracy loss (<1%)
        """
        print("\\n=== Model Quantization ===\\n")
        
        import torch
        import torch.nn as nn
        import time
        
        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(100, 10)
            
            def forward(self, x):
                return self.fc(x)
        
        # Create model
        model_fp32 = SimpleModel()
        model_fp32.eval()
        
        # Quantize to INT8
        model_int8 = torch.quantization.quantize_dynamic(
            model_fp32,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Compare sizes
        import os
        import tempfile
        
        # Save FP32
        fp32_path = tempfile.mktemp()
        torch.save(model_fp32.state_dict(), fp32_path)
        fp32_size_mb = os.path.getsize(fp32_path) / 1024 / 1024
        
        # Save INT8
        int8_path = tempfile.mktemp()
        torch.save(model_int8.state_dict(), int8_path)
        int8_size_mb = os.path.getsize(int8_path) / 1024 / 1024
        
        print(f"FP32 model size: {fp32_size_mb:.3f} MB")
        print(f"INT8 model size: {int8_size_mb:.3f} MB")
        print(f"Size reduction: {fp32_size_mb / int8_size_mb:.1f}x\\n")
        
        # Compare speed
        x = torch.randn(1, 100)
        
        # FP32 speed
        start = time.time()
        for _ in range(1000):
            model_fp32(x)
        fp32_time_ms = (time.time() - start) / 1000 * 1000
        
        # INT8 speed
        start = time.time()
        for _ in range(1000):
            model_int8(x)
        int8_time_ms = (time.time() - start) / 1000 * 1000
        
        print(f"FP32 inference: {fp32_time_ms:.3f} ms/prediction")
        print(f"INT8 inference: {int8_time_ms:.3f} ms/prediction")
        print(f"Speedup: {fp32_time_ms / int8_time_ms:.1f}x")
        
        # Cleanup
        os.remove(fp32_path)
        os.remove(int8_path)
    
    def feature_caching_strategy(self):
        """
        Cache expensive features
        """
        print("\\n=== Feature Caching Strategy ===\\n")
        
        print("Expensive Features (cache in Redis):")
        print("  - User historical stats (30-day aggregates)")
        print("  - Graph features (network connections)")
        print("  - Embeddings (pre-computed)")
        print("  TTL: 1 hour (refresh periodically)")
        
        print("\\nCheap Features (compute real-time):")
        print("  - Current transaction amount")
        print("  - Time of day")
        print("  - Simple ratios")
        
        print("\\nLatency Breakdown:")
        print("  Without caching:")
        print("    - Feature computation: 80ms")
        print("    - Model inference: 10ms")
        print("    - Total: 90ms")
        
        print("\\n  With caching:")
        print("    - Redis lookup: 2ms")
        print("    - Compute cheap features: 1ms")
        print("    - Model inference: 10ms")
        print("    - Total: 13ms")
        print("\\n  → 6.9x speedup!")
    
    def batching_strategy(self):
        """
        Batching for throughput
        """
        print("\\n=== Batching Strategy ===\\n")
        
        print("Trade-off:")
        print("  - Larger batches: Higher throughput, higher latency")
        print("  - Smaller batches: Lower latency, lower throughput")
        
        print("\\nStrategy:")
        print("  - Collect requests for 10ms")
        print("  - Predict on batch")
        print("  - Return results")
        
        print("\\nExample:")
        print("  Single prediction: 5ms each")
        print("  Batch of 32: 15ms total → 0.47ms per prediction")
        print("  But: +10ms waiting for batch")
        print("  Effective latency: 10-15ms (still good for many use cases)")


# Run optimization examples
optimizer = LatencyOptimization()
optimizer.model_quantization_example()
optimizer.feature_caching_strategy()
optimizer.batching_strategy()
\`\`\`

---

## Online Learning

### Continual Model Updates

\`\`\`python
"""
Online Learning: Update model continuously
"""

class OnlineLearning:
    """
    Update model with each new data point
    
    Benefits:
    - Always up-to-date
    - Adapt to distribution shifts quickly
    
    Challenges:
    - Catastrophic forgetting
    - Noisy updates
    - Computational cost
    """
    
    def __init__(self, model):
        self.model = model
        self.samples_seen = 0
    
    def incremental_learning(self, X_stream, y_stream):
        """
        Update model incrementally
        
        Uses: Scikit-learn's partial_fit
        """
        print("\\n=== Online Learning ===\\n")
        
        from sklearn.linear_model import SGDClassifier
        import numpy as np
        
        # Initialize online model
        online_model = SGDClassifier(loss='log_loss', random_state=42)
        
        # Initial fit (need at least one batch with all classes)
        initial_batch_size = 100
        X_initial = X_stream[:initial_batch_size]
        y_initial = y_stream[:initial_batch_size]
        
        online_model.partial_fit(X_initial, y_initial, classes=np.unique(y_stream))
        
        print(f"Initial training on {initial_batch_size} samples")
        
        # Stream remaining data
        batch_size = 32
        n_updates = 0
        
        for i in range(initial_batch_size, len(X_stream), batch_size):
            X_batch = X_stream[i:i+batch_size]
            y_batch = y_stream[i:i+batch_size]
            
            if len(X_batch) == 0:
                break
            
            # Incremental update
            online_model.partial_fit(X_batch, y_batch)
            
            n_updates += 1
            self.samples_seen += len(X_batch)
            
            if n_updates % 10 == 0:
                # Evaluate
                accuracy = online_model.score(X_batch, y_batch)
                print(f"Update {n_updates}: Samples seen: {self.samples_seen}, "
                      f"Recent accuracy: {accuracy:.3f}")
        
        print(f"\\n✓ Online learning complete: {n_updates} updates, "
              f"{self.samples_seen} samples")
        
        return online_model
    
    def windowed_training(self, X_stream, y_stream, window_size=1000):
        """
        Train on sliding window (forget old data)
        
        Helps with concept drift
        """
        print("\\n=== Windowed Training ===\\n")
        
        print(f"Window size: {window_size} samples")
        print("Strategy: Keep only recent data, retrain periodically")
        
        from collections import deque
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Buffer
        X_buffer = deque(maxlen=window_size)
        y_buffer = deque(maxlen=window_size)
        
        # Process stream
        retrain_frequency = 100  # Retrain every 100 samples
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        for i, (x, y) in enumerate(zip(X_stream, y_stream)):
            # Add to buffer
            X_buffer.append(x)
            y_buffer.append(y)
            
            # Retrain periodically
            if i % retrain_frequency == 0 and len(X_buffer) >= 100:
                X_train = np.array(X_buffer)
                y_train = np.array(y_buffer)
                
                model.fit(X_train, y_train)
                
                accuracy = model.score(X_train, y_train)
                print(f"Retrained at sample {i}: accuracy {accuracy:.3f}")
        
        print(f"\\n✓ Processed {len(X_stream)} samples with {len(X_stream) // retrain_frequency} retrains")
        
        return model


# Example: Online learning
from sklearn.datasets import make_classification
import numpy as np

# Generate stream of data
X_stream, y_stream = make_classification(
    n_samples=1000,
    n_features=20,
    random_state=42
)

# Online learning
online_learner = OnlineLearning(None)
online_model = online_learner.incremental_learning(X_stream, y_stream)

# Windowed training
windowed_model = online_learner.windowed_training(X_stream, y_stream, window_size=500)
\`\`\`

---

## Concept Drift Detection

### Detecting Distribution Shifts

\`\`\`python
"""
Concept Drift Detection
"""

class ConceptDriftDetector:
    """
    Detect when data distribution changes
    
    Types of drift:
    1. Covariate shift: P(X) changes, P(Y|X) stable
    2. Prior shift: P(Y) changes
    3. Concept drift: P(Y|X) changes
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reference_window = None
    
    def detect_covariate_shift(self, X_reference, X_current):
        """
        Detect shift in feature distribution
        
        Method: Kolmogorov-Smirnov test
        """
        print("\\n=== Covariate Shift Detection ===\\n")
        
        from scipy import stats
        import numpy as np
        
        # Test each feature
        drifted_features = []
        
        for feature_idx in range(X_reference.shape[1]):
            ref_feature = X_reference[:, feature_idx]
            cur_feature = X_current[:, feature_idx]
            
            # KS test
            statistic, p_value = stats.ks_2samp(ref_feature, cur_feature)
            
            # Drift if p < 0.05
            if p_value < 0.05:
                drifted_features.append((feature_idx, p_value, statistic))
        
        if drifted_features:
            print(f"🚨 Drift detected in {len(drifted_features)} features:")
            for feat_idx, p_val, stat in drifted_features:
                print(f"  Feature {feat_idx}: p={p_val:.4f}, KS={stat:.4f}")
            print("\\n→ Recommend: Retrain model with recent data")
        else:
            print("✓ No significant drift detected")
        
        return len(drifted_features) > 0
    
    def detect_prediction_drift(self, y_pred_reference, y_pred_current):
        """
        Detect shift in predictions
        
        Simpler than monitoring features
        """
        print("\\n=== Prediction Drift Detection ===\\n")
        
        import numpy as np
        
        # Compare distributions
        ref_mean = np.mean(y_pred_reference)
        cur_mean = np.mean(y_pred_current)
        
        ref_std = np.std(y_pred_reference)
        cur_std = np.std(y_pred_current)
        
        print(f"Reference predictions: mean={ref_mean:.3f}, std={ref_std:.3f}")
        print(f"Current predictions: mean={cur_mean:.3f}, std={cur_std:.3f}")
        
        # Simple threshold: >20% change in mean
        mean_change = abs(cur_mean - ref_mean) / (ref_mean + 1e-8)
        
        if mean_change > 0.20:
            print(f"\\n🚨 Prediction drift detected: {mean_change*100:.1f}% change")
            print("→ Recommend: Investigate and retrain")
            return True
        else:
            print(f"\\n✓ No significant drift ({mean_change*100:.1f}% change)")
            return False
    
    def detect_performance_degradation(self, y_true_recent, y_pred_recent, baseline_accuracy):
        """
        Detect drop in model performance
        
        Most direct signal
        """
        print("\\n=== Performance Degradation Detection ===\\n")
        
        from sklearn.metrics import accuracy_score
        
        current_accuracy = accuracy_score(y_true_recent, y_pred_recent)
        
        print(f"Baseline accuracy: {baseline_accuracy:.3f}")
        print(f"Current accuracy: {current_accuracy:.3f}")
        
        accuracy_drop = baseline_accuracy - current_accuracy
        
        if accuracy_drop > 0.05:  # >5% drop
            print(f"\\n🚨 Performance degradation detected: -{accuracy_drop*100:.1f}%")
            print("→ Action: Retrain immediately")
            return True
        else:
            print(f"\\n✓ Performance stable (drop: -{accuracy_drop*100:.1f}%)")
            return False


# Example: Drift detection
import numpy as np
from sklearn.datasets import make_classification

# Reference data
X_reference, y_reference = make_classification(n_samples=1000, n_features=20, random_state=42)

# Current data with drift (shift mean of first feature)
X_current, y_current = make_classification(n_samples=1000, n_features=20, random_state=43)
X_current[:, 0] += 2.0  # Introduce drift

# Train model on reference
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_reference, y_reference)

baseline_accuracy = model.score(X_reference, y_reference)

# Predictions
y_pred_reference = model.predict(X_reference)
y_pred_current = model.predict(X_current)

# Detect drift
detector = ConceptDriftDetector()

detector.detect_covariate_shift(X_reference[:100], X_current[:100])
detector.detect_prediction_drift(y_pred_reference, y_pred_current)
detector.detect_performance_degradation(y_current, y_pred_current, baseline_accuracy)
\`\`\`

---

## Real-Time Feature Engineering

### Feature Stores

\`\`\`python
"""
Real-Time Feature Store
"""

class RealTimeFeatureStore:
    """
    Serve features with low latency
    
    Requirements:
    - <10ms feature lookup
    - Consistency (training = serving)
    - Freshness (real-time updates)
    """
    
    def __init__(self):
        # Simulated Redis
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def compute_and_cache_features(self, user_id, historical_data):
        """
        Pre-compute expensive features
        """
        import numpy as np
        
        # Expensive aggregations
        features = {
            "user_30d_transactions": len(historical_data),
            "user_30d_total_spend": sum(historical_data),
            "user_30d_avg_transaction": np.mean(historical_data),
            "user_30d_max_transaction": max(historical_data) if historical_data else 0
        }
        
        # Cache with TTL
        self.cache[user_id] = features
        
        return features
    
    def get_features(self, user_id, current_transaction):
        """
        Get features for real-time prediction
        
        Mix of cached and real-time features
        """
        # Try cache first
        if user_id in self.cache:
            cached_features = self.cache[user_id]
            self.hit_count += 1
        else:
            # Cache miss: compute (expensive)
            cached_features = {
                "user_30d_transactions": 0,
                "user_30d_total_spend": 0,
                "user_30d_avg_transaction": 0,
                "user_30d_max_transaction": 0
            }
            self.miss_count += 1
        
        # Add real-time features
        features = {
            **cached_features,
            "current_amount": current_transaction['amount'],
            "hour_of_day": current_transaction['hour'],
            "amount_vs_avg_ratio": (
                current_transaction['amount'] / (cached_features['user_30d_avg_transaction'] + 1)
            )
        }
        
        return features
    
    def get_cache_stats(self):
        """
        Cache performance
        """
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "hits": self.hit_count,
            "misses": self.miss_count
        }


# Example: Feature store
import random

feature_store = RealTimeFeatureStore()

# Pre-populate cache for some users
for user_id in range(1000, 1100):
    historical = [random.uniform(10, 500) for _ in range(random.randint(5, 50))]
    feature_store.compute_and_cache_features(user_id, historical)

print("\\n=== Real-Time Feature Store ===\\n")
print("Pre-populated cache with 100 users\\n")

# Simulate real-time requests
for i in range(20):
    user_id = random.randint(1000, 1150)  # Some cache misses
    transaction = {
        "amount": random.uniform(10, 1000),
        "hour": random.randint(0, 23)
    }
    
    features = feature_store.get_features(user_id, transaction)
    
    print(f"Request {i+1}: User {user_id}, Amount \${transaction['amount']: .2f}")
print(f"  Features: 30d_transactions={features['user_30d_transactions']}, "
          f"amount_vs_avg={features['amount_vs_avg_ratio']:.2f}")

# Cache stats
stats = feature_store.get_cache_stats()
print(f"\\n=== Cache Performance ===")
print(f"Hit rate: {stats['hit_rate']*100:.1f}%")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
\`\`\`

---

## Key Takeaways

1. **Streaming Pipelines**: Kafka for real-time data ingestion
2. **Low Latency**: Model quantization, feature caching, batching
3. **Online Learning**: Incremental updates for fresh models
4. **Drift Detection**: Monitor distribution shifts and retrain
5. **Feature Store**: Cache expensive features, <10ms lookups

**Real-Time ML Checklist**:
- ✅ Stream processing (Kafka/Flink)
- ✅ Model optimization (quantization, pruning)
- ✅ Feature caching (Redis)
- ✅ Drift monitoring (automated detection)
- ✅ Online learning (continuous updates)
- ✅ Low-latency serving (<100ms P99)
- ✅ Fallback strategies (if model fails)

**Latency Budget Example (Trading)**:
- Network: 1ms
- Feature lookup: 2ms (Redis)
- Model inference: 5ms (quantized XGBoost)
- Post-processing: 1ms
- Response: 1ms
- **Total: 10ms** ✓

**Next Steps**: We'll explore LLM Production Systems with unique challenges like token streaming and cost management.
`,
};
