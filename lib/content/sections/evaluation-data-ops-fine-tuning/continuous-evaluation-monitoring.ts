/**
 * Continuous Evaluation & Monitoring Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const continuousEvaluationMonitoring = {
  id: 'continuous-evaluation-monitoring',
  title: 'Continuous Evaluation & Monitoring',
  content: `# Continuous Evaluation & Monitoring

Master monitoring AI systems in production and detecting performance degradation over time.

## Overview: Why Continuous Evaluation?

**Your model doesn't stop learning after deployment—the world changes.**

### What Changes:
- 📊 **Data Distribution**: User queries shift over time
- 🌐 **World Knowledge**: New events, facts, trends
- 🐛 **Edge Cases**: Users find new failure modes
- 💰 **User Expectations**: Standards increase
- ⚡ **Performance**: Latency, costs drift

### Without Monitoring:
- ❌ Silent degradation
- ❌ User churn from poor experience
- ❌ Missed opportunities for improvement
- ❌ No visibility into what's working

## Monitoring Framework

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import deque
import time

@dataclass
class Inference:
    """Single model inference."""
    id: str
    timestamp: float
    input: str
    output: str
    model_version: str
    latency_ms: float
    cost: float
    user_feedback: Optional[float] = None  # 1-5 rating
    automatic_score: Optional[float] = None

class ContinuousMonitor:
    """Monitor model performance in production."""
    
    def __init__(
        self,
        window_size: int = 1000,  # Rolling window of inferences
        alert_threshold: float = 0.15  # 15% degradation triggers alert
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Rolling windows
        self.recent_inferences = deque (maxlen=window_size)
        self.baseline_metrics: Optional[Dict[str, float]] = None
    
    def log_inference (self, inference: Inference):
        """Log inference for monitoring."""
        self.recent_inferences.append (inference)
    
    def set_baseline (self):
        """Set baseline metrics from current window."""
        if len (self.recent_inferences) < 100:
            print("Need at least 100 inferences for baseline")
            return
        
        self.baseline_metrics = self._calculate_metrics (list (self.recent_inferences))
        print(f"✅ Baseline set: {self.baseline_metrics}")
    
    def check_for_degradation (self) -> Dict[str, Any]:
        """Check if performance has degraded."""
        
        if not self.baseline_metrics:
            return {'error': 'No baseline set'}
        
        if len (self.recent_inferences) < 100:
            return {'error': 'Insufficient data'}
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics (list (self.recent_inferences))
        
        # Compare to baseline
        alerts = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics[metric]
            
            # Check for degradation
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value
                
                if degradation > self.alert_threshold:
                    alerts.append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': current_value,
                        'degradation_pct': degradation * 100
                    })
        
        return {
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': current_metrics,
            'alerts': alerts,
            'needs_attention': len (alerts) > 0
        }
    
    def _calculate_metrics (self, inferences: List[Inference]) -> Dict[str, float]:
        """Calculate metrics from inferences."""
        
        # User feedback score
        feedbacks = [inf.user_feedback for inf in inferences if inf.user_feedback]
        avg_feedback = sum (feedbacks) / len (feedbacks) if feedbacks else 0
        
        # Automatic scores
        auto_scores = [inf.automatic_score for inf in inferences if inf.automatic_score]
        avg_auto_score = sum (auto_scores) / len (auto_scores) if auto_scores else 0
        
        # Latency
        latencies = [inf.latency_ms for inf in inferences]
        avg_latency = sum (latencies) / len (latencies)
        p95_latency = sorted (latencies)[int (len (latencies) * 0.95)]
        
        # Cost
        costs = [inf.cost for inf in inferences]
        avg_cost = sum (costs) / len (costs)
        
        return {
            'avg_user_feedback': avg_feedback,
            'avg_automatic_score': avg_auto_score,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'avg_cost': avg_cost
        }
    
    def generate_report (self) -> str:
        """Generate monitoring report."""
        
        check = self.check_for_degradation()
        
        if 'error' in check:
            return f"Error: {check['error']}"
        
        report = "=== Continuous Monitoring Report ===\\n\\n"
        
        report += "Current Metrics:\\n"
        for metric, value in check['current_metrics'].items():
            report += f"  {metric}: {value:.4f}\\n"
        
        report += "\\n"
        
        if check['alerts']:
            report += "⚠️  ALERTS:\\n"
            for alert in check['alerts']:
                report += f"  - {alert['metric']}: {alert['degradation_pct']:.1f}% degradation\\n"
                report += f"    Baseline: {alert['baseline']:.4f} → Current: {alert['current']:.4f}\\n"
        else:
            report += "✅ No performance degradation detected\\n"
        
        return report

# Usage
monitor = ContinuousMonitor (window_size=1000)

# Log inferences as they happen
for inference in production_inferences:
    monitor.log_inference (inference)

# Set baseline after initial period
if len (monitor.recent_inferences) >= 1000:
    monitor.set_baseline()

# Periodically check for degradation
check = monitor.check_for_degradation()
if check.get('needs_attention'):
    print("⚠️  Performance degradation detected!")
    print(monitor.generate_report())
    # Send alert to team
\`\`\`

## Drift Detection

\`\`\`python
class DriftDetector:
    """Detect distribution shift in inputs/outputs."""
    
    def __init__(self):
        self.baseline_distribution = None
    
    def set_baseline_distribution(
        self,
        texts: List[str]
    ):
        """Set baseline text distribution."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode (texts)
        
        # Store statistics
        self.baseline_distribution = {
            'mean': embeddings.mean (axis=0),
            'std': embeddings.std (axis=0),
            'embeddings': embeddings
        }
    
    def detect_drift(
        self,
        current_texts: List[str],
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Detect if current distribution differs from baseline."""
        
        if not self.baseline_distribution:
            return {'error': 'No baseline set'}
        
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import cosine
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        current_embeddings = model.encode (current_texts)
        
        # Compare distributions
        baseline_mean = self.baseline_distribution['mean']
        current_mean = current_embeddings.mean (axis=0)
        
        # Distance between means
        distance = cosine (baseline_mean, current_mean)
        
        drift_detected = distance > threshold
        
        return {
            'drift_score': distance,
            'drift_detected': drift_detected,
            'threshold': threshold,
            'recommendation': 'Retrain model or update prompts' if drift_detected else 'No action needed'
        }

# Usage
drift_detector = DriftDetector()

# Set baseline from initial production data
drift_detector.set_baseline_distribution (initial_user_queries)

# Periodically check for drift
current_queries = get_recent_queries (last_n_days=7)
drift = drift_detector.detect_drift (current_queries)

if drift['drift_detected']:
    print(f"⚠️  Distribution drift detected!")
    print(f"Drift score: {drift['drift_score']:.3f}")
    print(f"Recommendation: {drift['recommendation']}")
\`\`\`

## Automated Re-Evaluation

\`\`\`python
import asyncio
from datetime import datetime, timedelta

class AutomatedEvaluator:
    """Automatically re-evaluate model on test set."""
    
    def __init__(
        self,
        test_dataset: List[Dict],
        model_fn: callable,
        evaluation_metrics: List[callable]
    ):
        self.test_dataset = test_dataset
        self.model_fn = model_fn
        self.evaluation_metrics = evaluation_metrics
        
        self.evaluation_history: List[Dict] = []
    
    async def run_evaluation (self) -> Dict[str, Any]:
        """Run full evaluation on test set."""
        
        results = []
        
        for example in self.test_dataset:
            output = await self.model_fn (example['input'])
            
            scores = {}
            for metric in self.evaluation_metrics:
                score = metric (output, example['expected_output'])
                scores[metric.__name__] = score
            
            results.append (scores)
        
        # Aggregate
        aggregated = {}
        for metric_name in results[0].keys():
            values = [r[metric_name] for r in results]
            aggregated[metric_name] = sum (values) / len (values)
        
        # Store in history
        evaluation_record = {
            'timestamp': datetime.now(),
            'metrics': aggregated
        }
        self.evaluation_history.append (evaluation_record)
        
        return aggregated
    
    async def schedule_periodic_evaluation(
        self,
        interval_hours: int = 24
    ):
        """Run evaluation every N hours."""
        
        print(f"Starting periodic evaluation (every {interval_hours}h)")
        
        while True:
            print(f"\\n[{datetime.now()}] Running scheduled evaluation...")
            
            try:
                metrics = await self.run_evaluation()
                
                print("Results:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.2%}")
                
                # Check for degradation
                if len (self.evaluation_history) >= 2:
                    self._check_regression()
            
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
            
            # Wait until next evaluation
            await asyncio.sleep (interval_hours * 3600)
    
    def _check_regression (self):
        """Check if performance has regressed."""
        
        if len (self.evaluation_history) < 2:
            return
        
        baseline = self.evaluation_history[0]['metrics']
        latest = self.evaluation_history[-1]['metrics']
        
        for metric, baseline_value in baseline.items():
            latest_value = latest[metric]
            
            if baseline_value > 0:
                change = ((latest_value - baseline_value) / baseline_value) * 100
                
                if change < -10:  # >10% regression
                    print(f"\\n⚠️  REGRESSION DETECTED:")
                    print(f"  Metric: {metric}")
                    print(f"  Baseline: {baseline_value:.2%}")
                    print(f"  Current: {latest_value:.2%}")
                    print(f"  Change: {change:.1f}%")
                    
                    # Send alert
                    self._send_alert (metric, baseline_value, latest_value)
    
    def _send_alert (self, metric: str, baseline: float, current: float):
        """Send alert to team."""
        # Integration with Slack, PagerDuty, email, etc.
        pass

# Usage
evaluator = AutomatedEvaluator(
    test_dataset=my_test_set,
    model_fn=my_model_function,
    evaluation_metrics=[accuracy_metric, coherence_metric]
)

# Run once
metrics = await evaluator.run_evaluation()

# Or schedule periodic evaluation
await evaluator.schedule_periodic_evaluation (interval_hours=24)
\`\`\`

## Dashboard & Visualization

\`\`\`python
class MonitoringDashboard:
    """Create monitoring dashboard."""
    
    def __init__(self, monitor: ContinuousMonitor):
        self.monitor = monitor
    
    def generate_dashboard_data (self) -> Dict[str, Any]:
        """Generate data for dashboard."""
        
        if not self.monitor.recent_inferences:
            return {}
        
        # Time series data
        time_series = self._create_time_series()
        
        # Current stats
        current = self.monitor.check_for_degradation()
        
        # Top errors/failures
        errors = self._analyze_errors()
        
        return {
            'time_series': time_series,
            'current_metrics': current.get('current_metrics', {}),
            'alerts': current.get('alerts', []),
            'errors': errors
        }
    
    def _create_time_series (self) -> Dict[str, List]:
        """Create time series for plotting."""
        
        # Group by time buckets
        buckets = {}
        for inf in self.monitor.recent_inferences:
            # 1-hour buckets
            bucket = int (inf.timestamp / 3600) * 3600
            
            if bucket not in buckets:
                buckets[bucket] = []
            
            buckets[bucket].append (inf)
        
        # Calculate metrics per bucket
        times = []
        latencies = []
        feedbacks = []
        
        for timestamp in sorted (buckets.keys()):
            infs = buckets[timestamp]
            
            times.append (timestamp)
            latencies.append (sum (i.latency_ms for i in infs) / len (infs))
            
            fb = [i.user_feedback for i in infs if i.user_feedback]
            feedbacks.append (sum (fb) / len (fb) if fb else None)
        
        return {
            'timestamps': times,
            'avg_latency': latencies,
            'avg_feedback': feedbacks
        }
    
    def _analyze_errors (self) -> List[Dict]:
        """Analyze common error patterns."""
        
        # Identify low-scoring inferences
        poor_inferences = [
            inf for inf in self.monitor.recent_inferences
            if inf.user_feedback and inf.user_feedback < 3.0
        ]
        
        # Extract patterns
        # (In production: use clustering, topic modeling)
        
        return [
            {
                'pattern': 'Long inputs (>1000 tokens)',
                'frequency': 15,
                'avg_score': 2.3
            },
            {
                'pattern': 'Questions about recent events',
                'frequency': 23,
                'avg_score': 2.1
            }
        ]

# Usage
dashboard = MonitoringDashboard (monitor)
dashboard_data = dashboard.generate_dashboard_data()

# Render with Plotly, Streamlit, Grafana, etc.
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace (go.Scatter(
    x=dashboard_data['time_series']['timestamps'],
    y=dashboard_data['time_series']['avg_latency'],
    name='Latency'
))

fig.show()
\`\`\`

## Production Checklist

✅ **Monitoring Infrastructure**
- [ ] Logging all inferences
- [ ] Storing user feedback
- [ ] Calculating automatic scores
- [ ] Rolling window metrics

✅ **Baseline & Alerts**
- [ ] Baseline metrics set
- [ ] Alert thresholds configured
- [ ] Alert delivery (Slack, email, PagerDuty)
- [ ] On-call rotation defined

✅ **Drift Detection**
- [ ] Input distribution monitored
- [ ] Output distribution tracked
- [ ] Periodic drift checks
- [ ] Retraining triggers defined

✅ **Automated Evaluation**
- [ ] Test set maintained
- [ ] Periodic re-evaluation scheduled
- [ ] Regression detection enabled
- [ ] Results dashboard accessible

✅ **Incident Response**
- [ ] Runbook for degradation
- [ ] Rollback procedure
- [ ] Escalation path
- [ ] Post-mortem process

## Next Steps

You now understand continuous monitoring. Finally, learn:
- Building complete evaluation platforms
- Integrating all evaluation components
- End-to-end production workflows
`,
};
