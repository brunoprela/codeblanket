export const mlSystemDesignPrinciples = {
  title: 'ML System Design Principles',
  id: 'ml-system-design-principles',
  content: `
# ML System Design Principles

## Introduction

Building production ML systems is fundamentally different from training models in notebooks. **A model is just 5-10% of a complete ML system**—the remaining 90-95% is infrastructure, data pipelines, monitoring, and operational concerns.

This section covers the foundational principles for designing **scalable, reliable, and maintainable** machine learning systems that can operate in production environments, including real-time trading systems.

### What Makes ML Systems Different from Traditional Software

**Traditional Software**:
- Deterministic: same input → same output
- Logic explicitly coded
- Testing focuses on edge cases
- Debugging via stack traces
- Performance: latency, throughput

**ML Systems**:
- Probabilistic: same input → distribution of outputs
- Logic learned from data
- Testing requires statistical validation
- Debugging involves data quality, model drift, feature issues
- Performance: latency + **prediction quality** + **data freshness**

### Real-World ML System Complexity

\`\`\`python
"""
ML System Components (Google's estimate)
"""

# What people think ML systems look like:
ml_system_naive = {
    "model": 100  # 100% of effort
}

# What ML systems actually look like:
ml_system_reality = {
    "model_code": 5,  # Actual ML code
    "data_collection": 10,
    "data_verification": 5,
    "feature_extraction": 15,
    "model_serving": 10,
    "monitoring": 10,
    "configuration": 5,
    "infrastructure": 20,
    "testing": 10,
    "deployment": 10
}

print(f"Model code: {ml_system_reality['model_code']}%")
print(f"Everything else: {100 - ml_system_reality['model_code']}%")
\`\`\`

By the end of this section, you'll understand:
- How to translate business problems into ML problems
- System design principles for production ML
- Trade-offs between different architectural choices
- Requirements gathering for ML systems
- How to design end-to-end ML pipelines

---

## From Business Problem to ML Problem

### The ML System Design Process

**Step 1: Understand the Business Problem**

\`\`\`python
"""
Example: Design ML System for Algorithmic Trading
"""

business_problem = {
    "objective": "Generate alpha (excess returns) through algorithmic trading",
    "constraints": [
        "Low latency (<100ms for signal generation)",
        "High throughput (10,000+ predictions/second)",
        "Real-time data processing",
        "Risk management (max drawdown 15%)",
        "Regulatory compliance (audit trail)"
    ],
    "success_metrics": [
        "Sharpe ratio > 2.0",
        "Max drawdown < 15%",
        "Win rate > 55%",
        "99th percentile latency < 100ms"
    ]
}
\`\`\`

**Step 2: Translate to ML Problem**

\`\`\`python
"""
ML Problem Formulation
"""

ml_problem = {
    "task_type": "regression",  # Predict price change or returns
    "input": {
        "features": [
            "Technical indicators (50+ features)",
            "Order book imbalance",
            "News sentiment",
            "Market regime indicators"
        ],
        "data_sources": [
            "Real-time market data (WebSocket)",
            "Historical OHLCV",
            "News API",
            "Social media sentiment"
        ]
    },
    "output": {
        "prediction": "Expected return next 5 minutes",
        "confidence": "Model uncertainty",
        "features_importance": "Explainability"
    },
    "training": {
        "data_size": "5 years daily + 1 year minute-level",
        "update_frequency": "Retrain weekly, online learning daily",
        "validation": "Walk-forward with expanding window"
    },
    "serving": {
        "latency_requirement": "<50ms",
        "availability": "99.9%",
        "throughput": "10,000 predictions/sec"
    }
}
\`\`\`

### Requirements Gathering Framework

\`\`\`python
"""
ML System Requirements Template
"""

class MLSystemRequirements:
    """
    Comprehensive requirements for ML system design
    """
    
    def __init__(self, project_name):
        self.project_name = project_name
        self.requirements = {}
    
    def define_functional_requirements(self):
        """What the system must do"""
        return {
            "prediction_task": "What are we predicting?",
            "input_data": "What data do we need?",
            "output_format": "What format for predictions?",
            "user_interface": "How do users interact?",
            "integrations": "What systems to integrate with?"
        }
    
    def define_non_functional_requirements(self):
        """How well the system must perform"""
        return {
            "performance": {
                "latency": "P50, P95, P99 latency targets",
                "throughput": "Requests per second",
                "availability": "Uptime SLA (99.9%?)",
                "scalability": "Growth projections"
            },
            "data_requirements": {
                "volume": "Data size (GB, TB)",
                "velocity": "Update frequency",
                "variety": "Data types and sources",
                "quality": "Acceptable error rates"
            },
            "model_requirements": {
                "accuracy": "Minimum acceptable metrics",
                "freshness": "How stale can predictions be?",
                "explainability": "Do we need interpretability?",
                "bias": "Fairness constraints"
            },
            "operational": {
                "monitoring": "What to track",
                "alerting": "When to alert",
                "retraining": "Update frequency",
                "rollback": "Can we revert models?"
            },
            "compliance": {
                "data_privacy": "GDPR, CCPA",
                "auditability": "Logging requirements",
                "security": "Access control"
            }
        }
    
    def define_constraints(self):
        """System limitations"""
        return {
            "budget": "Infrastructure costs",
            "team": "Available expertise",
            "timeline": "Delivery deadline",
            "technology": "Required tools/platforms",
            "legacy": "Existing systems to work with"
        }
    
    def prioritize_requirements(self):
        """Must-have vs nice-to-have"""
        return {
            "must_have": [
                "Core prediction functionality",
                "Minimum acceptable accuracy",
                "Basic monitoring",
                "Data pipeline"
            ],
            "should_have": [
                "Advanced monitoring",
                "A/B testing",
                "Model explainability",
                "Automated retraining"
            ],
            "nice_to_have": [
                "Advanced UI",
                "Multi-model ensemble",
                "Real-time retraining",
                "Advanced analytics dashboard"
            ]
        }


# Example: Trading system requirements
trading_requirements = MLSystemRequirements("AlgoTrading System")

print("=== Trading System Requirements ===\\n")
print("Functional:")
for key, val in trading_requirements.define_functional_requirements().items():
    print(f"  {key}: {val}")

print("\\nNon-Functional:")
nfr = trading_requirements.define_non_functional_requirements()
print(f"  Latency: {nfr['performance']['latency']}")
print(f"  Accuracy: {nfr['model_requirements']['accuracy']}")
\`\`\`

---

## ML System Design Principles

### Principle 1: Start Simple, Iterate

\`\`\`python
"""
Progression from Simple to Complex
"""

# Phase 1: Simple baseline (Week 1)
baseline = {
    "model": "Linear regression",
    "features": "10 technical indicators",
    "serving": "Batch predictions daily",
    "monitoring": "Manual checks",
    "goal": "Establish baseline performance"
}

# Phase 2: Improved model (Month 1)
improved = {
    "model": "XGBoost",
    "features": "50+ engineered features",
    "serving": "Hourly batch predictions",
    "monitoring": "Automated metrics logging",
    "goal": "Beat baseline by 20%"
}

# Phase 3: Production system (Month 3)
production = {
    "model": "Ensemble of XGBoost + LightGBM",
    "features": "100+ features + real-time data",
    "serving": "Real-time predictions (<100ms)",
    "monitoring": "Full observability stack",
    "goal": "Deploy to production safely"
}

# Phase 4: Advanced system (Month 6+)
advanced = {
    "model": "Deep learning + LLM for sentiment",
    "features": "Multi-modal (price + text + alternative data)",
    "serving": "Sub-50ms with model optimization",
    "monitoring": "ML-specific drift detection",
    "goal": "Maximize Sharpe ratio"
}

def evaluate_complexity_tradeoff(phase):
    """
    Evaluate effort vs value
    """
    complexity_scores = {
        "baseline": {"effort": 1, "value": 3},
        "improved": {"effort": 3, "value": 5},
        "production": {"effort": 7, "value": 8},
        "advanced": {"effort": 10, "value": 9}
    }
    
    score = complexity_scores.get(phase, {"effort": 0, "value": 0})
    roi = score["value"] / score["effort"] if score["effort"] > 0 else 0
    
    return {
        "effort": score["effort"],
        "value": score["value"],
        "roi": roi
    }

for phase in ["baseline", "improved", "production", "advanced"]:
    result = evaluate_complexity_tradeoff(phase)
    print(f"{phase.capitalize()}: Effort={result['effort']}, "
          f"Value={result['value']}, ROI={result['roi']:.2f}")
\`\`\`

**Output**:
\`\`\`
Baseline: Effort=1, Value=3, ROI=3.00
Improved: Effort=3, Value=5, ROI=1.67
Production: Effort=7, Value=8, ROI=1.14
Advanced: Effort=10, Value=9, ROI=0.90
\`\`\`

**Lesson**: Baseline gives best ROI. Diminishing returns as complexity increases.

### Principle 2: Separation of Concerns

\`\`\`python
"""
Modular ML System Architecture
"""

# GOOD: Separated concerns
class MLPipeline:
    """
    Clean separation of responsibilities
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model = Model()
        self.predictor = Predictor()
        self.monitor = Monitor()
    
    def train(self, training_data):
        """Training pipeline"""
        # Each component handles one responsibility
        raw_data = self.data_loader.load(training_data)
        features = self.feature_engineer.transform(raw_data)
        self.model.fit(features)
        
        # Monitoring separate from training logic
        self.monitor.log_training_metrics(self.model.metrics)
    
    def predict(self, input_data):
        """Inference pipeline"""
        raw_data = self.data_loader.load(input_data)
        features = self.feature_engineer.transform(raw_data)
        predictions = self.predictor.predict(features, self.model)
        
        # Monitoring separate from prediction logic
        self.monitor.log_predictions(predictions)
        
        return predictions


# BAD: Everything mixed together
def train_and_predict_mixed(data, test_data):
    """
    ❌ Don't do this: mixed concerns
    """
    # Data loading, feature engineering, training, prediction all mixed
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    
    # Load data (should be separate)
    df = pd.read_csv(data)
    
    # Feature engineering (should be separate)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = calculate_rsi(df['close'])
    
    # Training (should be separate)
    model = RandomForestRegressor()
    model.fit(df[['sma_20', 'rsi']], df['target'])
    
    # Prediction (should be separate)
    test_df = pd.read_csv(test_data)
    test_df['sma_20'] = test_df['close'].rolling(20).mean()
    predictions = model.predict(test_df[['sma_20', 'rsi']])
    
    # Logging (should be separate)
    print(f"Predictions: {predictions}")
    
    return predictions
\`\`\`

### Principle 3: Data is First-Class Citizen

\`\`\`python
"""
Data-Centric ML System Design
"""

class DataCentricPipeline:
    """
    Data quality, versioning, and lineage as primary concerns
    """
    
    def __init__(self):
        self.data_version = "v1.0"
        self.data_schema = self.define_schema()
    
    def define_schema(self):
        """
        Explicit data schema (contract)
        """
        return {
            "features": {
                "price": {"type": "float", "range": [0, 1e6], "nullable": False},
                "volume": {"type": "int", "range": [0, 1e9], "nullable": False},
                "sentiment": {"type": "float", "range": [-1, 1], "nullable": True}
            },
            "target": {
                "returns": {"type": "float", "range": [-0.1, 0.1], "nullable": False}
            }
        }
    
    def validate_data(self, data):
        """
        Enforce data quality
        """
        issues = []
        
        for col, spec in self.data_schema["features"].items():
            if col not in data.columns:
                issues.append(f"Missing column: {col}")
                continue
            
            # Type check
            if data[col].dtype != spec["type"]:
                issues.append(f"{col}: wrong type {data[col].dtype} vs {spec['type']}")
            
            # Range check
            if spec["range"]:
                min_val, max_val = spec["range"]
                out_of_range = ((data[col] < min_val) | (data[col] > max_val)).sum()
                if out_of_range > 0:
                    issues.append(f"{col}: {out_of_range} values out of range")
            
            # Null check
            if not spec["nullable"] and data[col].isnull().any():
                issues.append(f"{col}: contains nulls but shouldn't")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "data_version": self.data_version
        }
    
    def version_data(self, data, metadata):
        """
        Version data for reproducibility
        """
        import hashlib
        import json
        
        # Create data fingerprint
        data_hash = hashlib.sha256(
            data.to_json().encode()
        ).hexdigest()[:16]
        
        version_info = {
            "version": self.data_version,
            "hash": data_hash,
            "shape": data.shape,
            "timestamp": metadata.get("timestamp"),
            "source": metadata.get("source"),
            "transformations": metadata.get("transformations", [])
        }
        
        return version_info


# Usage
pipeline = DataCentricPipeline()

# Validate data
import pandas as pd
import numpy as np

sample_data = pd.DataFrame({
    "price": np.random.uniform(100, 200, 1000),
    "volume": np.random.randint(1e6, 1e7, 1000),
    "sentiment": np.random.uniform(-1, 1, 1000),
    "returns": np.random.normal(0, 0.02, 1000)
})

validation = pipeline.validate_data(sample_data)
print(f"Data valid: {validation['valid']}")
if not validation['valid']:
    print("Issues:")
    for issue in validation['issues']:
        print(f"  - {issue}")

# Version data
version_info = pipeline.version_data(
    sample_data,
    metadata={
        "timestamp": "2024-01-01",
        "source": "market_data_api",
        "transformations": ["remove_outliers", "fill_missing"]
    }
)

print(f"\\nData version: {version_info}")
\`\`\`

### Principle 4: Fail Fast, Fail Safe

\`\`\`python
"""
Error Handling and Graceful Degradation
"""

class RobustMLService:
    """
    Production-ready ML service with error handling
    """
    
    def __init__(self, primary_model, fallback_model):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.error_count = 0
        self.max_errors = 10
    
    def predict_with_fallback(self, features):
        """
        Try primary model, fallback if fails
        """
        try:
            # Validate input
            self._validate_input(features)
            
            # Try primary model
            prediction = self.primary_model.predict(features)
            
            # Validate output
            self._validate_output(prediction)
            
            return {
                "prediction": prediction,
                "model": "primary",
                "confidence": "high"
            }
        
        except Exception as e:
            self.error_count += 1
            
            # Log error
            print(f"❌ Primary model failed: {e}")
            
            # Circuit breaker: stop if too many errors
            if self.error_count > self.max_errors:
                raise RuntimeError("Circuit breaker: too many errors")
            
            # Try fallback model
            try:
                prediction = self.fallback_model.predict(features)
                
                return {
                    "prediction": prediction,
                    "model": "fallback",
                    "confidence": "low"
                }
            
            except Exception as e2:
                print(f"❌ Fallback model failed: {e2}")
                
                # Last resort: return safe default
                return {
                    "prediction": self._safe_default(features),
                    "model": "default",
                    "confidence": "none"
                }
    
    def _validate_input(self, features):
        """Input validation"""
        if features is None:
            raise ValueError("Features cannot be None")
        
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {features.shape}")
        
        if np.isnan(features).any():
            raise ValueError("Features contain NaN values")
    
    def _validate_output(self, prediction):
        """Output validation"""
        if prediction is None:
            raise ValueError("Prediction is None")
        
        if np.isnan(prediction).any():
            raise ValueError("Prediction contains NaN")
        
        # Sanity check: returns should be reasonable
        if np.abs(prediction).max() > 0.5:
            raise ValueError(f"Prediction out of range: {prediction}")
    
    def _safe_default(self, features):
        """
        Safe default prediction when all else fails
        """
        # Return zero (no prediction) or last known good prediction
        return np.zeros((features.shape[0], 1))


# Example usage
import numpy as np
from sklearn.linear_model import LinearRegression

# Create models
primary = LinearRegression().fit(
    np.random.randn(100, 10),
    np.random.randn(100, 1)
)

fallback = LinearRegression().fit(
    np.random.randn(50, 10),
    np.random.randn(50, 1)
)

service = RobustMLService(primary, fallback)

# Normal case
features = np.random.randn(5, 10)
result = service.predict_with_fallback(features)
print(f"\\nPrediction: {result['prediction'][:3]}")
print(f"Model used: {result['model']}")
print(f"Confidence: {result['confidence']}")

# Error case: invalid input
try:
    result = service.predict_with_fallback(None)
except Exception as e:
    print(f"\\nError handled: {e}")
\`\`\`

---

## ML System Architecture Patterns

### Pattern 1: Batch Prediction Architecture

\`\`\`python
"""
Batch Prediction System (Daily/Hourly predictions)
"""

class BatchPredictionSystem:
    """
    For non-real-time use cases (e.g., daily trading signals)
    
    Pros:
    - Simple to implement
    - Easy to debug
    - Cost-effective
    - Can use complex models
    
    Cons:
    - High latency
    - Not suitable for real-time
    """
    
    def __init__(self):
        self.scheduler = None  # e.g., Airflow, cron
        self.model = None
        self.storage = None  # e.g., S3, database
    
    def daily_batch_job(self):
        """
        Run once per day
        """
        print("Starting batch job...")
        
        # 1. Load data
        data = self.load_data()
        print(f"Loaded {len(data)} records")
        
        # 2. Preprocess
        features = self.preprocess(data)
        
        # 3. Predict (can take minutes)
        predictions = self.model.predict(features)
        
        # 4. Post-process
        results = self.post_process(predictions, data)
        
        # 5. Store results
        self.store_predictions(results)
        
        print("Batch job complete")
        
        return results
    
    def load_data(self):
        """Load data from database"""
        # In practice: query database
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "price": [150, 2800, 300]
        })
    
    def preprocess(self, data):
        """Feature engineering"""
        # Can afford expensive computations
        return data  # Simplified
    
    def post_process(self, predictions, data):
        """Add metadata, filtering"""
        data['prediction'] = predictions
        data['timestamp'] = pd.Timestamp.now()
        return data
    
    def store_predictions(self, results):
        """Store for later use"""
        # In practice: write to database, S3, etc.
        print(f"Stored {len(results)} predictions")


# Usage
batch_system = BatchPredictionSystem()
# In production: scheduled by Airflow daily at 6am
# results = batch_system.daily_batch_job()
\`\`\`

### Pattern 2: Real-Time Prediction Architecture

\`\`\`python
"""
Real-Time Prediction System (<100ms latency)
"""

import time
from typing import Dict, Any

class RealTimePredictionSystem:
    """
    For low-latency use cases (e.g., HFT, real-time trading)
    
    Pros:
    - Low latency
    - Fresh predictions
    - React to market changes
    
    Cons:
    - Complex infrastructure
    - Limited model complexity
    - Higher costs
    """
    
    def __init__(self, model, feature_cache):
        self.model = model
        self.feature_cache = feature_cache  # Redis, in-memory
        self.latency_budget_ms = 50
    
    def predict_realtime(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle real-time prediction request
        
        Latency budget: 50ms
        """
        start_time = time.time()
        
        try:
            # 1. Validate (< 1ms)
            self._validate_request(request)
            
            # 2. Fetch cached features (< 5ms)
            features = self._get_cached_features(request)
            
            # 3. Predict (< 30ms)
            prediction = self.model.predict(features)
            
            # 4. Post-process (< 5ms)
            result = self._format_response(prediction, request)
            
            # 5. Log async (don't wait)
            self._log_async(request, result)
            
            # Check latency
            latency_ms = (time.time() - start_time) * 1000
            result['latency_ms'] = latency_ms
            
            if latency_ms > self.latency_budget_ms:
                print(f"⚠️  Latency exceeded: {latency_ms:.2f}ms")
            
            return result
        
        except Exception as e:
            # Fast failure
            return {
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    def _validate_request(self, request):
        """Quick validation"""
        required = ['symbol', 'timestamp']
        for field in required:
            if field not in request:
                raise ValueError(f"Missing field: {field}")
    
    def _get_cached_features(self, request):
        """
        Fetch pre-computed features from cache
        
        Critical: Don't compute features in real-time!
        """
        # In practice: Redis lookup
        symbol = request['symbol']
        
        # Simulated cache hit
        features = {
            'sma_20': 150.5,
            'rsi': 65.3,
            'volume_ratio': 1.2
        }
        
        return np.array([[features['sma_20'], features['rsi'], features['volume_ratio']]])
    
    def _format_response(self, prediction, request):
        """Format response"""
        return {
            'symbol': request['symbol'],
            'prediction': float(prediction[0]),
            'timestamp': time.time()
        }
    
    def _log_async(self, request, result):
        """
        Async logging (don't block)
        """
        # In practice: push to queue (Kafka, SQS)
        pass


# Usage
import numpy as np
from sklearn.linear_model import LinearRegression

# Train simple model
model = LinearRegression().fit(
    np.random.randn(1000, 3),
    np.random.randn(1000)
)

realtime_system = RealTimePredictionSystem(model, feature_cache={})

# Simulate request
request = {
    "symbol": "AAPL",
    "timestamp": time.time()
}

result = realtime_system.predict_realtime(request)
print(f"\\nPrediction: {result['prediction']:.4f}")
print(f"Latency: {result['latency_ms']:.2f}ms")
\`\`\`

### Pattern 3: Hybrid Architecture

\`\`\`python
"""
Hybrid: Batch + Real-Time
"""

class HybridMLSystem:
    """
    Batch for expensive features, real-time for fast inference
    
    Example: Trading system
    - Batch: Compute complex features daily (fundamentals, sentiment)
    - Real-time: Use cached features + live prices for fast predictions
    """
    
    def __init__(self):
        self.batch_system = BatchPredictionSystem()
        self.realtime_system = RealTimePredictionSystem(None, None)
    
    def daily_feature_computation(self):
        """
        Batch job: Compute expensive features
        Run daily at 6am
        """
        print("Computing expensive features...")
        
        # Compute features that take minutes
        features = {
            "AAPL": {
                "pe_ratio": 25.3,
                "sentiment_7d": 0.65,
                "analyst_rating": 4.2,
                # ... 100+ features
            }
        }
        
        # Store in cache for real-time access
        self._update_feature_cache(features)
        
        print("Features cached for real-time use")
    
    def realtime_prediction(self, symbol, live_price):
        """
        Real-time: Use cached features + live data
        """
        # Fast: just lookup cached features
        cached_features = self._get_cached_features(symbol)
        
        # Fast: add live data
        live_features = {
            "price": live_price,
            "price_change": live_price - cached_features.get("prev_close", live_price)
        }
        
        # Fast: predict
        features = {**cached_features, **live_features}
        prediction = self._fast_model_predict(features)
        
        return prediction
    
    def _update_feature_cache(self, features):
        """Update Redis/cache"""
        pass
    
    def _get_cached_features(self, symbol):
        """Get from cache"""
        return {"prev_close": 150.0}
    
    def _fast_model_predict(self, features):
        """Fast model inference"""
        return 0.005  # Predicted return


# Usage
hybrid = HybridMLSystem()

# Once per day: batch job
# hybrid.daily_feature_computation()

# Many times per second: real-time
prediction = hybrid.realtime_prediction("AAPL", live_price=151.25)
print(f"\\nReal-time prediction: {prediction:.4f}")
\`\`\`

---

## Trade-Offs in ML System Design

### Latency vs Accuracy

\`\`\`python
"""
Model Complexity vs Latency Trade-off
"""

import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Prepare data
X = np.random.randn(10000, 50)
y = np.random.randn(10000)

X_test = np.random.randn(1, 50)

# Train models
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6)
}

for name, model in models.items():
    model.fit(X, y)

# Compare latency
print("\\n=== Model Comparison ===")
print(f"{'Model':<15} {'Latency (ms)':<15} {'Suitable For'}")
print("-" * 50)

for name, model in models.items():
    # Warm-up
    _ = model.predict(X_test)
    
    # Measure latency
    start = time.time()
    for _ in range(100):
        _ = model.predict(X_test)
    latency_ms = (time.time() - start) / 100 * 1000
    
    # Determine use case
    if latency_ms < 1:
        use_case = "Real-time (HFT)"
    elif latency_ms < 10:
        use_case = "Real-time (trading)"
    elif latency_ms < 100:
        use_case = "Near real-time"
    else:
        use_case = "Batch only"
    
    print(f"{name:<15} {latency_ms:<15.2f} {use_case}")

print("\\n📊 Trade-off: Linear is 10-100x faster but less accurate")
print("   Solution: Use Linear for real-time, XGBoost for batch")
\`\`\`

### Cost vs Performance

\`\`\`python
"""
Infrastructure Cost Analysis
"""

def estimate_monthly_cost(architecture: str):
    """
    Estimate infrastructure costs
    """
    costs = {
        "simple_batch": {
            "compute": 100,  # Run daily for 1 hour
            "storage": 50,
            "total": 150
        },
        "realtime_cpu": {
            "compute": 1000,  # Always-on server
            "storage": 100,
            "cache": 200,  # Redis
            "total": 1300
        },
        "realtime_gpu": {
            "compute": 5000,  # GPU instance
            "storage": 100,
            "cache": 200,
            "total": 5300
        }
    }
    
    return costs.get(architecture, {"total": 0})

# Compare architectures
print("\\n=== Monthly Infrastructure Costs ===")
for arch in ["simple_batch", "realtime_cpu", "realtime_gpu"]:
    cost = estimate_monthly_cost(arch)
    print(f"{arch}: \${cost['total']}/month")

print("\\n💡 Tip: Start with batch, upgrade to real-time only if needed")
\`\`\`

---

## Key Takeaways

1. **ML System ≠ Just a Model**: Model is 5-10%, infrastructure is 90-95%
2. **Requirements First**: Clearly define functional, non-functional, and constraints
3. **Start Simple**: Baseline model with simple infrastructure, iterate
4. **Separation of Concerns**: Modular design (data, features, model, serving, monitoring)
5. **Data-Centric**: Treat data as first-class citizen (versioning, validation, lineage)
6. **Fail Safe**: Error handling, fallbacks, circuit breakers
7. **Trade-offs**: Balance latency vs accuracy, cost vs performance
8. **Architecture Patterns**:
   - **Batch**: Simple, cost-effective, higher latency
   - **Real-time**: Complex, expensive, low latency
   - **Hybrid**: Best of both worlds

**Next Steps**: With design principles established, we'll dive into data engineering, model training, and deployment in subsequent sections.
`,
};
