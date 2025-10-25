export const modelMonitoring = {
  title: 'Model Monitoring',
  id: 'model-monitoring',
  content: `
# Model Monitoring

## Introduction

**"You can't manage what you don't measure."**

In production, models **degrade over time**. Data distributions shift, user behavior changes, markets evolve. Without monitoring, you won't know when your model stops working until it's too late.

**Why Models Fail in Production**:
- **Data drift**: Input distribution changes
- **Concept drift**: Relationship between features and target changes
- **Data quality issues**: Missing values, outliers, corrupted data
- **System issues**: Latency spikes, service failures
- **Adversarial behavior**: Users gaming the system

This section covers comprehensive monitoring strategies to keep models healthy in production.

### What to Monitor

\`\`\`
Input Data → Model → Predictions → Business Metrics
     ↓          ↓          ↓              ↓
  Drift?    Latency?   Quality?      Impact?
  Schema?   Errors?    Confidence?   Revenue?
  Volume?   Memory?    Distribution? Conversions?
\`\`\`

By the end of this section, you'll understand:
- Data drift detection
- Model performance monitoring
- System health metrics
- Alerting strategies
- Monitoring tools and dashboards

---

## Data Drift Detection

### Statistical Tests for Drift

\`\`\`python
"""
Data Drift Detection
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import warnings

class DataDriftDetector:
    """
    Detect data drift using statistical tests
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Args:
            reference_data: Training/baseline data
        """
        self.reference_data = reference_data
        self.reference_stats = self._compute_statistics (reference_data)
    
    def _compute_statistics (self, data: pd.DataFrame) -> Dict:
        """
        Compute baseline statistics
        """
        stats_dict = {}
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype (data[col]):
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'q25': data[col].quantile(0.25),
                    'q50': data[col].quantile(0.50),
                    'q75': data[col].quantile(0.75)
                }
        
        return stats_dict
    
    def kolmogorov_smirnov_test(
        self,
        current_data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Kolmogorov-Smirnov test for distribution drift
        
        Tests if two samples come from the same distribution
        """
        results = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype (current_data[col]):
                continue
            
            # KS test
            statistic, pvalue = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            # Drift detected if p-value < alpha
            drift_detected = pvalue < alpha
            
            results[col] = {
                'test': 'KS',
                'statistic': float (statistic),
                'pvalue': float (pvalue),
                'drift_detected': drift_detected,
                'threshold': alpha
            }
        
        return results
    
    def population_stability_index(
        self,
        current_data: pd.DataFrame,
        bins: int = 10
    ) -> Dict[str, float]:
        """
        Population Stability Index (PSI)
        
        PSI < 0.1: No significant drift
        0.1 ≤ PSI < 0.2: Moderate drift
        PSI ≥ 0.2: Significant drift
        """
        psi_scores = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype (current_data[col]):
                continue
            
            # Create bins based on reference data
            ref_col = self.reference_data[col].dropna()
            cur_col = current_data[col].dropna()
            
            # Define bin edges
            bin_edges = np.linspace(
                ref_col.min(),
                ref_col.max(),
                bins + 1
            )
            
            # Calculate distributions
            ref_dist, _ = np.histogram (ref_col, bins=bin_edges)
            cur_dist, _ = np.histogram (cur_col, bins=bin_edges)
            
            # Convert to proportions
            ref_dist = ref_dist / ref_dist.sum()
            cur_dist = cur_dist / cur_dist.sum()
            
            # Avoid division by zero
            ref_dist = np.where (ref_dist == 0, 0.0001, ref_dist)
            cur_dist = np.where (cur_dist == 0, 0.0001, cur_dist)
            
            # Calculate PSI
            psi = np.sum((cur_dist - ref_dist) * np.log (cur_dist / ref_dist))
            
            psi_scores[col] = float (psi)
        
        return psi_scores
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        methods: list = ['ks', 'psi']
    ) -> Dict:
        """
        Run all drift detection methods
        """
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'sample_size': {
                'reference': len (self.reference_data),
                'current': len (current_data)
            },
            'drift_detected': False,
            'features_with_drift': [],
            'tests': {}
        }
        
        # KS test
        if 'ks' in methods:
            ks_results = self.kolmogorov_smirnov_test (current_data)
            results['tests']['ks'] = ks_results
            
            # Check for drift
            for feature, test_result in ks_results.items():
                if test_result['drift_detected']:
                    results['features_with_drift'].append({
                        'feature': feature,
                        'method': 'KS',
                        'pvalue': test_result['pvalue']
                    })
        
        # PSI
        if 'psi' in methods:
            psi_scores = self.population_stability_index (current_data)
            results['tests']['psi'] = psi_scores
            
            # Check for drift (PSI >= 0.2)
            for feature, psi in psi_scores.items():
                if psi >= 0.2:
                    results['features_with_drift'].append({
                        'feature': feature,
                        'method': 'PSI',
                        'score': psi
                    })
        
        # Overall drift flag
        results['drift_detected'] = len (results['features_with_drift']) > 0
        
        return results


# Example usage
np.random.seed(42)

# Reference data (training)
reference = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 10000),
    'feature_2': np.random.normal(5, 2, 10000),
    'feature_3': np.random.exponential(1, 10000)
})

# Current data (with drift in feature_1)
current = pd.DataFrame({
    'feature_1': np.random.normal(0.5, 1.2, 1000),  # Drift: mean shifted, std increased
    'feature_2': np.random.normal(5, 2, 1000),      # No drift
    'feature_3': np.random.exponential(1, 1000)     # No drift
})

# Detect drift
detector = DataDriftDetector (reference)
drift_report = detector.detect_drift (current, methods=['ks', 'psi'])

print("=== Drift Detection Report ===")
print(f"Drift detected: {drift_report['drift_detected']}")
print(f"\\nFeatures with drift:")
for item in drift_report['features_with_drift']:
    print(f"  - {item['feature']} ({item['method']}): {item.get('pvalue', item.get('score')):.4f}")

print(f"\\nPSI Scores:")
for feature, psi in drift_report['tests']['psi'].items():
    status = "⚠️ DRIFT" if psi >= 0.2 else "✓ OK"
    print(f"  {feature}: {psi:.4f} {status}")
\`\`\`

### Feature Drift Monitoring

\`\`\`python
"""
Real-time Feature Drift Monitoring
"""

from collections import deque
from datetime import datetime, timedelta
import threading
import time

class FeatureDriftMonitor:
    """
    Monitor feature drift in real-time
    """
    
    def __init__(self, reference_stats: Dict, window_size: int = 1000):
        self.reference_stats = reference_stats
        self.window_size = window_size
        
        # Sliding windows for each feature
        self.feature_windows = {
            feature: deque (maxlen=window_size)
            for feature in reference_stats.keys()
        }
        
        self.alerts = []
        self.monitoring = False
    
    def add_sample (self, features: Dict[str, float]):
        """
        Add new sample to monitoring windows
        """
        for feature, value in features.items():
            if feature in self.feature_windows:
                self.feature_windows[feature].append (value)
    
    def check_drift (self) -> Dict[str, bool]:
        """
        Check current drift status
        """
        drift_status = {}
        
        for feature, window in self.feature_windows.items():
            if len (window) < 100:  # Need minimum samples
                continue
            
            # Current statistics
            current_mean = np.mean (window)
            current_std = np.std (window)
            
            # Reference statistics
            ref_mean = self.reference_stats[feature]['mean']
            ref_std = self.reference_stats[feature]['std']
            
            # Z-score for mean shift
            z_score = abs (current_mean - ref_mean) / ref_std if ref_std > 0 else 0
            
            # Drift if z-score > 3 (3 sigma)
            drift_detected = z_score > 3
            
            drift_status[feature] = {
                'drift': drift_detected,
                'z_score': z_score,
                'current_mean': current_mean,
                'reference_mean': ref_mean
            }
            
            # Alert if drift
            if drift_detected:
                alert = {
                    'timestamp': datetime.now(),
                    'feature': feature,
                    'z_score': z_score,
                    'message': f"Drift detected in {feature}: z-score={z_score:.2f}"
                }
                self.alerts.append (alert)
                print(f"🚨 ALERT: {alert['message']}")
        
        return drift_status
    
    def start_monitoring (self, check_interval: int = 60):
        """
        Start background monitoring
        
        Args:
            check_interval: Seconds between checks
        """
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                self.check_drift()
                time.sleep (check_interval)
        
        thread = threading.Thread (target=monitor_loop, daemon=True)
        thread.start()
        
        print(f"✓ Monitoring started (check every {check_interval}s)")
    
    def stop_monitoring (self):
        """Stop monitoring"""
        self.monitoring = False
        print("✓ Monitoring stopped")
    
    def get_recent_alerts (self, hours: int = 24) -> list:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta (hours=hours)
        
        recent = [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff
        ]
        
        return recent


# Example usage
reference_stats = {
    'feature_1': {'mean': 0.0, 'std': 1.0},
    'feature_2': {'mean': 5.0, 'std': 2.0}
}

monitor = FeatureDriftMonitor (reference_stats, window_size=1000)

# Simulate incoming data
for i in range(2000):
    # Normal data first
    if i < 1000:
        sample = {
            'feature_1': np.random.normal(0, 1),
            'feature_2': np.random.normal(5, 2)
        }
    # Drifted data after
    else:
        sample = {
            'feature_1': np.random.normal(0.5, 1.5),  # Drift!
            'feature_2': np.random.normal(5, 2)
        }
    
    monitor.add_sample (sample)
    
    # Check every 100 samples
    if i % 100 == 0:
        drift_status = monitor.check_drift()
        
        if i == 1100:  # Should detect drift here
            print(f"\\nSample {i}:")
            for feature, status in drift_status.items():
                print(f"  {feature}: drift={status['drift']}, z={status['z_score']:.2f}")
\`\`\`

---

## Model Performance Monitoring

### Prediction Quality Monitoring

\`\`\`python
"""
Monitor Model Prediction Quality
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from collections import defaultdict

class ModelPerformanceMonitor:
    """
    Monitor model performance over time
    """
    
    def __init__(self, model_type='regression', baseline_metrics=None):
        self.model_type = model_type
        self.baseline_metrics = baseline_metrics or {}
        
        # Store predictions and actuals for evaluation
        self.predictions = []
        self.actuals = []
        
        # Performance over time
        self.performance_history = defaultdict (list)
        
        # Prediction distribution tracking
        self.prediction_stats = {
            'min': [],
            'max': [],
            'mean': [],
            'std': []
        }
    
    def log_prediction (self, prediction: float, actual: float = None, metadata: Dict = None):
        """
        Log a prediction (and actual if available)
        """
        self.predictions.append({
            'timestamp': pd.Timestamp.now(),
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        })
        
        if actual is not None:
            self.actuals.append (actual)
    
    def compute_metrics (self, window: int = 1000) -> Dict:
        """
        Compute metrics on recent predictions
        """
        if len (self.predictions) < 10:
            return {}
        
        # Get recent predictions with actuals
        recent = self.predictions[-window:]
        recent_with_actuals = [p for p in recent if p['actual'] is not None]
        
        if len (recent_with_actuals) < 10:
            return {}
        
        preds = [p['prediction'] for p in recent_with_actuals]
        actuals = [p['actual'] for p in recent_with_actuals]
        
        metrics = {}
        
        if self.model_type == 'regression':
            metrics['rmse'] = np.sqrt (mean_squared_error (actuals, preds))
            metrics['mae'] = np.mean (np.abs (np.array (actuals) - np.array (preds)))
            metrics['r2'] = r2_score (actuals, preds)
        
        elif self.model_type == 'classification':
            metrics['accuracy'] = accuracy_score (actuals, preds)
        
        # Prediction distribution
        metrics['pred_mean'] = np.mean (preds)
        metrics['pred_std'] = np.std (preds)
        metrics['pred_min'] = np.min (preds)
        metrics['pred_max'] = np.max (preds)
        
        return metrics
    
    def detect_degradation (self, current_metrics: Dict) -> Dict:
        """
        Detect model degradation
        """
        if not self.baseline_metrics:
            return {'degraded': False, 'reason': 'No baseline set'}
        
        degradation = {
            'degraded': False,
            'metrics_worse': [],
            'severity': 'none'
        }
        
        # Check each metric
        for metric, baseline_value in self.baseline_metrics.items():
            if metric not in current_metrics:
                continue
            
            current_value = current_metrics[metric]
            
            # For metrics where lower is better (RMSE, MAE)
            if metric in ['rmse', 'mae']:
                degradation_pct = (current_value - baseline_value) / baseline_value * 100
                
                if degradation_pct > 20:  # 20% worse
                    degradation['degraded'] = True
                    degradation['metrics_worse'].append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_pct': degradation_pct
                    })
            
            # For metrics where higher is better (R2, accuracy)
            elif metric in ['r2', 'accuracy']:
                degradation_pct = (baseline_value - current_value) / baseline_value * 100
                
                if degradation_pct > 10:  # 10% worse
                    degradation['degraded'] = True
                    degradation['metrics_worse'].append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_pct': degradation_pct
                    })
        
        # Determine severity
        if degradation['degraded']:
            max_degradation = max (m['degradation_pct'] for m in degradation['metrics_worse'])
            
            if max_degradation > 50:
                degradation['severity'] = 'critical'
            elif max_degradation > 30:
                degradation['severity'] = 'high'
            else:
                degradation['severity'] = 'medium'
        
        return degradation
    
    def generate_report (self) -> str:
        """
        Generate monitoring report
        """
        current_metrics = self.compute_metrics()
        degradation = self.detect_degradation (current_metrics)
        
        report = "\\n=== Model Performance Report ===\\n"
        report += f"Total predictions: {len (self.predictions)}\\n"
        report += f"Predictions with actuals: {len([p for p in self.predictions if p['actual'] is not None])}\\n"
        
        report += "\\nCurrent Metrics:\\n"
        for metric, value in current_metrics.items():
            report += f"  {metric}: {value:.4f}\\n"
        
        report += "\\nDegradation Check:\\n"
        report += f"  Degraded: {degradation['degraded']}\\n"
        
        if degradation['degraded']:
            report += f"  Severity: {degradation['severity'].upper()}\\n"
            report += "  Worse metrics:\\n"
            for m in degradation['metrics_worse']:
                report += f"    - {m['metric']}: {m['baseline']:.4f} → {m['current']:.4f} ({m['degradation_pct']:.1f}% worse)\\n"
        
        return report


# Example usage
monitor = ModelPerformanceMonitor(
    model_type='regression',
    baseline_metrics={'rmse': 0.1, 'r2': 0.85}
)

# Simulate predictions (good model first, degraded later)
for i in range(1000):
    # Generate prediction
    true_value = np.random.randn()
    
    if i < 500:
        # Good model
        prediction = true_value + np.random.normal(0, 0.1)
    else:
        # Degraded model
        prediction = true_value + np.random.normal(0, 0.3)
    
    monitor.log_prediction (prediction, actual=true_value)

# Check performance
report = monitor.generate_report()
print(report)
\`\`\`

---

## System Health Monitoring

### Latency and Throughput Monitoring

\`\`\`python
"""
System Health Monitoring
"""

import time
from collections import deque
from typing import Dict
import psutil

class SystemHealthMonitor:
    """
    Monitor system health metrics
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Metric windows
        self.latencies = deque (maxlen=window_size)
        self.timestamps = deque (maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        
        # Thresholds
        self.latency_threshold_p95 = 100  # ms
        self.latency_threshold_p99 = 200  # ms
        self.error_rate_threshold = 0.01  # 1%
    
    def log_request (self, latency_ms: float, error: bool = False):
        """
        Log a request
        """
        self.latencies.append (latency_ms)
        self.timestamps.append (time.time())
        self.total_requests += 1
        
        if error:
            self.error_count += 1
    
    def get_latency_stats (self) -> Dict:
        """
        Get latency statistics
        """
        if not self.latencies:
            return {}
        
        latencies_array = np.array (self.latencies)
        
        return {
            'p50': np.percentile (latencies_array, 50),
            'p95': np.percentile (latencies_array, 95),
            'p99': np.percentile (latencies_array, 99),
            'mean': np.mean (latencies_array),
            'max': np.max (latencies_array),
            'min': np.min (latencies_array)
        }
    
    def get_throughput (self, window_seconds: int = 60) -> float:
        """
        Get requests per second
        """
        if len (self.timestamps) < 2:
            return 0.0
        
        # Count requests in last window_seconds
        cutoff = time.time() - window_seconds
        recent_requests = sum(1 for ts in self.timestamps if ts > cutoff)
        
        return recent_requests / window_seconds
    
    def get_error_rate (self) -> float:
        """
        Get error rate
        """
        if self.total_requests == 0:
            return 0.0
        
        return self.error_count / self.total_requests
    
    def get_system_resources (self) -> Dict:
        """
        Get system resource usage
        """
        return {
            'cpu_percent': psutil.cpu_percent (interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def check_health (self) -> Dict:
        """
        Check overall system health
        """
        latency_stats = self.get_latency_stats()
        throughput = self.get_throughput()
        error_rate = self.get_error_rate()
        resources = self.get_system_resources()
        
        health = {
            'healthy': True,
            'issues': [],
            'metrics': {
                'latency': latency_stats,
                'throughput': throughput,
                'error_rate': error_rate,
                'resources': resources
            }
        }
        
        # Check latency
        if latency_stats and latency_stats['p95'] > self.latency_threshold_p95:
            health['healthy'] = False
            health['issues'].append({
                'type': 'latency',
                'severity': 'warning',
                'message': f"P95 latency {latency_stats['p95']:.1f}ms exceeds threshold {self.latency_threshold_p95}ms"
            })
        
        if latency_stats and latency_stats['p99'] > self.latency_threshold_p99:
            health['healthy'] = False
            health['issues'].append({
                'type': 'latency',
                'severity': 'critical',
                'message': f"P99 latency {latency_stats['p99']:.1f}ms exceeds threshold {self.latency_threshold_p99}ms"
            })
        
        # Check error rate
        if error_rate > self.error_rate_threshold:
            health['healthy'] = False
            health['issues'].append({
                'type': 'errors',
                'severity': 'critical',
                'message': f"Error rate {error_rate*100:.2f}% exceeds threshold {self.error_rate_threshold*100:.2f}%"
            })
        
        # Check CPU
        if resources['cpu_percent'] > 90:
            health['healthy'] = False
            health['issues'].append({
                'type': 'resource',
                'severity': 'warning',
                'message': f"CPU usage {resources['cpu_percent']:.1f}% is very high"
            })
        
        # Check memory
        if resources['memory_percent'] > 90:
            health['healthy'] = False
            health['issues'].append({
                'type': 'resource',
                'severity': 'critical',
                'message': f"Memory usage {resources['memory_percent']:.1f}% is very high"
            })
        
        return health


# Example usage
system_monitor = SystemHealthMonitor()

# Simulate requests
for i in range(1000):
    # Most requests are fast
    if i < 900:
        latency = np.random.uniform(10, 50)
        error = False
    # Some slow requests
    elif i < 980:
        latency = np.random.uniform(80, 120)
        error = False
    # Some very slow requests
    else:
        latency = np.random.uniform(150, 250)
        error = np.random.random() < 0.1  # 10% error rate
    
    system_monitor.log_request (latency, error)

# Check health
health = system_monitor.check_health()

print("\\n=== System Health Report ===")
print(f"Healthy: {health['healthy']}")
print(f"\\nLatency:")
for metric, value in health['metrics']['latency'].items():
    print(f"  {metric}: {value:.2f}ms")
print(f"\\nThroughput: {health['metrics']['throughput']:.1f} req/s")
print(f"Error rate: {health['metrics']['error_rate']*100:.2f}%")

if not health['healthy']:
    print(f"\\n⚠️ Issues detected:")
    for issue in health['issues']:
        print(f"  [{issue['severity'].upper()}] {issue['message']}")
\`\`\`

---

## Alerting System

### Alert Manager

\`\`\`python
"""
Comprehensive Alerting System
"""

from enum import Enum
from typing import List, Callable
import smtplib
from email.mime.text import MIMEText

class AlertSeverity(Enum):
    INFO = 1
    WARNING = 2
    CRITICAL = 3

class Alert:
    """Alert object"""
    
    def __init__(self, severity: AlertSeverity, title: str, message: str, metadata: Dict = None):
        self.severity = severity
        self.title = title
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = pd.Timestamp.now()
    
    def __str__(self):
        return f"[{self.severity.name}] {self.title}: {self.message}"


class AlertManager:
    """
    Manage alerts and notifications
    """
    
    def __init__(self):
        self.alerts = []
        self.notification_channels = []
    
    def add_channel (self, channel: Callable):
        """
        Add notification channel
        
        Args:
            channel: Function that takes Alert object
        """
        self.notification_channels.append (channel)
    
    def trigger_alert (self, alert: Alert):
        """
        Trigger an alert
        """
        self.alerts.append (alert)
        
        # Send to all channels
        for channel in self.notification_channels:
            try:
                channel (alert)
            except Exception as e:
                print(f"Failed to send alert via {channel.__name__}: {e}")
    
    def get_recent_alerts (self, hours: int = 24, severity: AlertSeverity = None) -> List[Alert]:
        """
        Get recent alerts
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta (hours=hours)
        
        recent = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff
        ]
        
        if severity:
            recent = [a for a in recent if a.severity == severity]
        
        return recent


# Notification channels
def slack_notification (alert: Alert):
    """
    Send alert to Slack
    """
    # In production: use Slack API
    print(f"📱 SLACK: {alert}")

def email_notification (alert: Alert):
    """
    Send alert via email
    """
    # In production: use SMTP
    print(f"📧 EMAIL: {alert}")

def pagerduty_notification (alert: Alert):
    """
    Send to PagerDuty (for critical alerts)
    """
    if alert.severity == AlertSeverity.CRITICAL:
        print(f"🚨 PAGERDUTY: {alert}")


# Example usage
alert_manager = AlertManager()

# Add notification channels
alert_manager.add_channel (slack_notification)
alert_manager.add_channel (email_notification)
alert_manager.add_channel (pagerduty_notification)

# Trigger alerts
alert_manager.trigger_alert(Alert(
    severity=AlertSeverity.WARNING,
    title="High Latency",
    message="P95 latency exceeded 100ms",
    metadata={'p95': 125, 'threshold': 100}
))

alert_manager.trigger_alert(Alert(
    severity=AlertSeverity.CRITICAL,
    title="Data Drift Detected",
    message="Feature drift detected in 3 features",
    metadata={'features': ['feature_1', 'feature_2', 'feature_3']}
))

# Get recent critical alerts
critical_alerts = alert_manager.get_recent_alerts (hours=24, severity=AlertSeverity.CRITICAL)
print(f"\\nCritical alerts (last 24h): {len (critical_alerts)}")
\`\`\`

---

## Monitoring Dashboard

### Prometheus + Grafana Setup

\`\`\`python
"""
Export Metrics to Prometheus
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

prediction_value = Histogram(
    'model_prediction_value',
    'Prediction values distribution',
    ['model_version']
)

model_drift_score = Gauge(
    'model_drift_score',
    'Data drift score (PSI)',
    ['feature']
)

error_rate = Gauge(
    'model_error_rate',
    'Model error rate',
    ['model_version']
)


def predict_with_metrics (features, model_version='v1.0'):
    """
    Make prediction with metric tracking
    """
    start = time.time()
    
    try:
        # Make prediction
        prediction = np.random.randn()  # Simulated
        
        # Record metrics
        prediction_counter.labels(
            model_version=model_version,
            status='success'
        ).inc()
        
        prediction_latency.labels(
            model_version=model_version
        ).observe (time.time() - start)
        
        prediction_value.labels(
            model_version=model_version
        ).observe (prediction)
        
        return prediction
    
    except Exception as e:
        prediction_counter.labels(
            model_version=model_version,
            status='error'
        ).inc()
        
        raise


# Start Prometheus metrics server
# start_http_server(8000)

# Simulate predictions
for i in range(100):
    predict_with_metrics([0.5] * 10)
    
    # Update drift scores
    model_drift_score.labels (feature='feature_1').set(0.15)
    model_drift_score.labels (feature='feature_2').set(0.08)

print("Metrics exported to http://localhost:8000/metrics")
print("Configure Prometheus to scrape this endpoint")
\`\`\`

---

## Key Takeaways

1. **Data Drift**: Monitor input distribution changes (KS test, PSI)
2. **Performance Drift**: Track model metrics over time (RMSE, accuracy)
3. **System Health**: Monitor latency, throughput, errors, resources
4. **Alerting**: Multi-channel alerts (Slack, email, PagerDuty)
5. **Dashboards**: Prometheus + Grafana for visualization

**Trading-Specific**:
- Monitor prediction distribution (should be stable)
- Track Sharpe ratio and drawdown in real-time
- Alert on unusual trading patterns
- Monitor order execution latency

**Next Steps**: With monitoring in place, we'll cover A/B testing for safely deploying model updates.
`,
};
