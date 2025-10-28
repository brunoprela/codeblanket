export const dataQualityValidation = {
    title: 'Data Quality and Validation',
    id: 'data-quality-validation',
    content: `
# Data Quality and Validation

## Introduction

Data quality is mission-critical in algorithmic trading - bad data leads to bad decisions, which lead to financial losses. A single erroneous tick can trigger millions in incorrect trades, cause strategy malfunctions, or violate regulatory requirements. Production trading systems must implement comprehensive validation at every stage of the data pipeline.

**Why Data Quality Matters:**
- **Financial Impact**: Knight Capital lost $440M in 45 minutes due to bad orders triggered by software errors
- **Regulatory**: SEC requires audit trails - bad data = compliance violations
- **Strategy Performance**: Garbage in = garbage out. Strategies trained on bad data will fail in live trading
- **Reputation**: Providing bad data to clients destroys trust and business
- **Operational**: False signals from bad data waste time and resources

**Common Data Quality Issues in Production:**
- **Crossed Markets**: Bid > ask (market structure violation)
- **Price Spikes**: AAPL $150 → $1500 in 1ms (fat finger or bad data)
- **Missing Ticks**: Sequence gaps causing incomplete order books
- **Duplicate Ticks**: Same tick delivered multiple times
- **Stale Data**: Quotes not updating (feed disconnection)
- **Symbol Errors**: Wrong symbol mappings or invalid tickers
- **Timestamp Issues**: Out-of-order ticks, clock skew

This section covers real-time validation, anomaly detection, data quality metrics, and building production validation frameworks.

---

## Real-Time Validation Rules

### Basic Sanity Checks

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, List

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    severity: str  # 'critical', 'high', 'medium', 'low'

class RealTimeValidator:
    """Real-time market data validator"""
    
    def __init__(self):
        # Validation stats
        self.total_validated = 0
        self.total_errors = 0
        self.error_types = {}
        
        # Historical data for comparison
        self.last_quotes = {}  # symbol -> last_quote
        self.price_ranges = {}  # symbol -> (min, max) 24hr range
    
    def validate_quote(self, quote: dict) -> ValidationResult:
        """Validate a single quote"""
        errors = []
        warnings = []
        
        self.total_validated += 1
        
        # 1. Check for required fields
        required_fields = ['symbol', 'timestamp', 'bid_price', 'ask_price', 
                          'bid_size', 'ask_size']
        for field in required_fields:
            if field not in quote or quote[field] is None:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            self.total_errors += 1
            return ValidationResult(False, errors, warnings, 'critical')
        
        # 2. Check for crossed market (bid >= ask)
        bid = Decimal(str(quote['bid_price']))
        ask = Decimal(str(quote['ask_price']))
        
        if bid > ask:
            errors.append(f"Crossed market: bid={bid} > ask={ask}")
            self._record_error('crossed_market')
        elif bid == ask:
            warnings.append(f"Locked market: bid={bid} == ask={ask}")
        
        # 3. Check for negative or zero prices
        if bid <= 0:
            errors.append(f"Invalid bid price: {bid}")
            self._record_error('negative_price')
        if ask <= 0:
            errors.append(f"Invalid ask price: {ask}")
            self._record_error('negative_price')
        
        # 4. Check for negative or zero sizes
        if quote['bid_size'] <= 0:
            errors.append(f"Invalid bid size: {quote['bid_size']}")
        if quote['ask_size'] <= 0:
            errors.append(f"Invalid ask size: {quote['ask_size']}")
        
        # 5. Check spread (should be reasonable)
        spread = ask - bid
        mid = (bid + ask) / 2
        spread_bps = float(spread / mid) * 10000
        
        if spread_bps > 100:  # > 100 basis points
            warnings.append(f"Wide spread: {spread_bps:.1f} bps")
        
        # 6. Check for price spikes (compare to last quote)
        symbol = quote['symbol']
        if symbol in self.last_quotes:
            last_quote = self.last_quotes[symbol]
            last_mid = (last_quote['bid_price'] + last_quote['ask_price']) / 2
            current_mid = (bid + ask) / 2
            
            pct_change = abs(float((current_mid - last_mid) / last_mid)) * 100
            
            if pct_change > 10:  # 10% move
                errors.append(f"Price spike: \{pct_change:.1f}% move from $\{last_mid} to $\{current_mid}")
                self._record_error('price_spike')
            elif pct_change > 5:  # 5% move
                warnings.append(f"Large price move: {pct_change:.1f}%")
        
        # 7. Check timestamp (should be recent and monotonic)
        now = datetime.now()
        quote_time = quote['timestamp']
        
        if isinstance(quote_time, str):
            quote_time = datetime.fromisoformat(quote_time)
        
        age_seconds = (now - quote_time).total_seconds()
        
        if age_seconds < 0:
            errors.append(f"Future timestamp: {age_seconds:.1f}s in future")
        elif age_seconds > 60:
            warnings.append(f"Stale quote: {age_seconds:.1f}s old")
        
        # Update last quote for next validation
        self.last_quotes[symbol] = {
            'bid_price': bid,
            'ask_price': ask,
            'timestamp': quote_time
        }
        
        # Determine severity
        if errors:
            severity = 'critical' if 'crossed_market' in str(errors) else 'high'
            self.total_errors += 1
        else:
            severity = 'low' if warnings else 'none'
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, severity)
    
    def _record_error(self, error_type: str):
        """Record error for statistics"""
        if error_type not in self.error_types:
            self.error_types[error_type] = 0
        self.error_types[error_type] += 1
    
    def get_error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_validated == 0:
            return 0.0
        return float(self.total_errors) / float(self.total_validated)
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        return {
            'total_validated': self.total_validated,
            'total_errors': self.total_errors,
            'error_rate': self.get_error_rate(),
            'errors_by_type': self.error_types
        }

# Usage
validator = RealTimeValidator()

quote = {
    'symbol': 'AAPL',
    'timestamp': datetime.now(),
    'bid_price': 150.24,
    'ask_price': 150.26,
    'bid_size': 500,
    'ask_size': 300
}

result = validator.validate_quote(quote)
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")

# Check error stats periodically
stats = validator.get_error_stats()
print(f"Error rate: {stats['error_rate']*100:.2f}%")
\`\`\`

---

## Statistical Anomaly Detection

### Outlier Detection (Z-Score)

\`\`\`python
import numpy as np
from collections import deque

class AnomalyDetector:
    """Detect anomalies using statistical methods"""
    
    def __init__(self, window_size: int = 100, threshold_sigma: float = 3.0):
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        
        # Rolling windows per symbol
        self.price_windows = {}  # symbol -> deque of prices
        self.volume_windows = {}  # symbol -> deque of volumes
        
        # Anomaly counters
        self.anomalies_detected = 0
    
    def detect_price_anomaly(self, symbol: str, price: float) -> dict:
        """Detect if price is an outlier"""
        if symbol not in self.price_windows:
            self.price_windows[symbol] = deque(maxlen=self.window_size)
        
        window = self.price_windows[symbol]
        window.append(price)
        
        if len(window) < 30:  # Need minimum data points
            return {'is_anomaly': False, 'reason': 'insufficient_data'}
        
        # Calculate z-score
        mean = np.mean(window)
        std = np.std(window)
        
        if std == 0:  # Avoid division by zero
            return {'is_anomaly': False, 'reason': 'zero_variance'}
        
        z_score = abs((price - mean) / std)
        
        if z_score > self.threshold_sigma:
            self.anomalies_detected += 1
            return {
                'is_anomaly': True,
                'reason': 'outlier',
                'z_score': z_score,
                'mean': mean,
                'std': std,
                'threshold': self.threshold_sigma
            }
        
        return {'is_anomaly': False, 'z_score': z_score}
    
    def detect_volume_spike(self, symbol: str, volume: int) -> dict:
        """Detect unusual volume"""
        if symbol not in self.volume_windows:
            self.volume_windows[symbol] = deque(maxlen=self.window_size)
        
        window = self.volume_windows[symbol]
        window.append(volume)
        
        if len(window) < 30:
            return {'is_spike': False}
        
        # Volume spike: > 3x median
        median_volume = np.median(window)
        
        if volume > median_volume * 3:
            return {
                'is_spike': True,
                'volume': volume,
                'median': median_volume,
                'ratio': float(volume) / float(median_volume)
            }
        
        return {'is_spike': False}

# Usage
detector = AnomalyDetector(window_size=100, threshold_sigma=3.0)

# Check for price anomaly
result = detector.detect_price_anomaly('AAPL', 155.50)
if result['is_anomaly']:
    print(f"Price anomaly detected! Z-score: {result['z_score']:.2f}")
    print(f"Price: $155.50 vs Mean: \${result['mean']:.2f}(±\\$\{result['std']:.2f})")
\`\`\`

---

## Data Quality Metrics

### Quality Score Calculation

\`\`\`python
from datetime import datetime, timedelta

class DataQualityMetrics:
    """Track and calculate data quality metrics"""
    
    def __init__(self):
        # Metrics per symbol
        self.metrics = {}  # symbol -> metrics dict
        
        # Global metrics
        self.global_completeness = 1.0
        self.global_accuracy = 1.0
        self.global_timeliness = 1.0
    
    def update_metrics(self, symbol: str, quote: dict, validation_result):
        """Update quality metrics for a quote"""
        if symbol not in self.metrics:
            self.metrics[symbol] = {
                'total_quotes': 0,
                'valid_quotes': 0,
                'errors': 0,
                'warnings': 0,
                'avg_latency_ms': 0.0,
                'last_update': None,
                'gaps': 0
            }
        
        m = self.metrics[symbol]
        m['total_quotes'] += 1
        
        if validation_result.is_valid:
            m['valid_quotes'] += 1
        else:
            m['errors'] += len(validation_result.errors)
            m['warnings'] += len(validation_result.warnings)
        
        # Calculate latency (if quote has both timestamps)
        if 'exchange_timestamp' in quote and 'receive_timestamp' in quote:
            exchange_time = quote['exchange_timestamp']
            receive_time = quote['receive_timestamp']
            latency_ms = (receive_time - exchange_time).total_seconds() * 1000
            
            # Update rolling average
            alpha = 0.1  # Smoothing factor
            m['avg_latency_ms'] = (alpha * latency_ms + 
                                  (1 - alpha) * m['avg_latency_ms'])
        
        # Detect gaps
        if m['last_update']:
            gap_seconds = (quote['timestamp'] - m['last_update']).total_seconds()
            if gap_seconds > 10:  # 10 second gap
                m['gaps'] += 1
        
        m['last_update'] = quote['timestamp']
    
    def calculate_quality_score(self, symbol: str) -> dict:
        """Calculate overall quality score (0-100)"""
        if symbol not in self.metrics:
            return {'score': 0, 'components': {}}
        
        m = self.metrics[symbol]
        
        if m['total_quotes'] == 0:
            return {'score': 0, 'components': {}}
        
        # Component scores (0-100)
        
        # 1. Accuracy: % of valid quotes
        accuracy = (m['valid_quotes'] / m['total_quotes']) * 100
        
        # 2. Completeness: Inverse of gap ratio
        gap_ratio = m['gaps'] / max(m['total_quotes'], 1)
        completeness = max(0, (1 - gap_ratio) * 100)
        
        # 3. Timeliness: Based on latency (< 100ms = 100, > 1000ms = 0)
        latency_ms = m['avg_latency_ms']
        if latency_ms < 100:
            timeliness = 100
        elif latency_ms > 1000:
            timeliness = 0
        else:
            timeliness = 100 - ((latency_ms - 100) / 900) * 100
        
        # Overall score (weighted average)
        weights = {'accuracy': 0.5, 'completeness': 0.3, 'timeliness': 0.2}
        overall = (accuracy * weights['accuracy'] +
                  completeness * weights['completeness'] +
                  timeliness * weights['timeliness'])
        
        return {
            'score': overall,
            'components': {
                'accuracy': accuracy,
                'completeness': completeness,
                'timeliness': timeliness
            },
            'metrics': m
        }
    
    def get_quality_report(self) -> dict:
        """Generate comprehensive quality report"""
        report = {
            'timestamp': datetime.now(),
            'symbols': {}
        }
        
        for symbol in self.metrics:
            report['symbols'][symbol] = self.calculate_quality_score(symbol)
        
        # Calculate average quality across all symbols
        if report['symbols']:
            avg_score = sum(s['score'] for s in report['symbols'].values()) / len(report['symbols'])
            report['average_quality'] = avg_score
        else:
            report['average_quality'] = 0
        
        return report

# Usage
quality_metrics = DataQualityMetrics()

# Update metrics for each quote
validator = RealTimeValidator()
quote = {...}
validation_result = validator.validate_quote(quote)
quality_metrics.update_metrics('AAPL', quote, validation_result)

# Get quality score
score_result = quality_metrics.calculate_quality_score('AAPL')
print(f"Quality Score: {score_result['score']:.1f}/100")
print(f"  Accuracy: {score_result['components']['accuracy']:.1f}%")
print(f"  Completeness: {score_result['components']['completeness']:.1f}%")
print(f"  Timeliness: {score_result['components']['timeliness']:.1f}%")
\`\`\`

---

## Production Validation Framework

\`\`\`python
import asyncio
from typing import Callable, List

class ProductionValidationPipeline:
    """Complete validation pipeline for production"""
    
    def __init__(self):
        self.validator = RealTimeValidator()
        self.anomaly_detector = AnomalyDetector()
        self.quality_metrics = DataQualityMetrics()
        
        # Callbacks for different severity levels
        self.critical_callbacks: List[Callable] = []
        self.warning_callbacks: List[Callable] = []
        
        # Statistics
        self.processed_count = 0
        self.rejected_count = 0
    
    def register_critical_callback(self, callback: Callable):
        """Register callback for critical errors"""
        self.critical_callbacks.append(callback)
    
    def register_warning_callback(self, callback: Callable):
        """Register callback for warnings"""
        self.warning_callbacks.append(callback)
    
    async def validate_and_process(self, quote: dict) -> bool:
        """Validate quote through full pipeline"""
        self.processed_count += 1
        symbol = quote['symbol']
        
        # Step 1: Basic validation
        validation_result = self.validator.validate_quote(quote)
        
        # Step 2: Anomaly detection
        mid_price = (float(quote['bid_price']) + float(quote['ask_price'])) / 2
        anomaly_result = self.anomaly_detector.detect_price_anomaly(symbol, mid_price)
        
        # Step 3: Update quality metrics
        self.quality_metrics.update_metrics(symbol, quote, validation_result)
        
        # Step 4: Handle validation result
        if not validation_result.is_valid:
            self.rejected_count += 1
            
            # Execute critical callbacks
            for callback in self.critical_callbacks:
                await callback(quote, validation_result)
            
            return False  # Reject quote
        
        # Step 5: Handle anomalies (warnings only, don't reject)
        if anomaly_result.get('is_anomaly'):
            for callback in self.warning_callbacks:
                await callback(quote, anomaly_result)
        
        return True  # Accept quote
    
    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        rejection_rate = 0.0
        if self.processed_count > 0:
            rejection_rate = float(self.rejected_count) / float(self.processed_count)
        
        return {
            'processed': self.processed_count,
            'rejected': self.rejected_count,
            'rejection_rate': rejection_rate,
            'validator_stats': self.validator.get_error_stats(),
            'anomalies_detected': self.anomaly_detector.anomalies_detected
        }

# Usage
pipeline = ProductionValidationPipeline()

# Register alert callbacks
async def critical_alert(quote, result):
    print(f"CRITICAL: {quote['symbol']} - {result.errors}")
    # Send to PagerDuty, Slack, etc.

async def warning_alert(quote, result):
    print(f"WARNING: {quote['symbol']} anomaly detected")

pipeline.register_critical_callback(critical_alert)
pipeline.register_warning_callback(warning_alert)

# Process quotes
async def process_stream():
    for quote in quote_stream:
        is_valid = await pipeline.validate_and_process(quote)
        if is_valid:
            # Forward to strategies
            await distribute_quote(quote)
        else:
            # Log rejected quote
            logger.error(f"Rejected quote: {quote}")
\`\`\`

---

## Best Practices

1. **Validate early** - Catch errors at ingestion, not in strategies
2. **Multi-layer validation** - Basic + statistical + cross-reference
3. **Monitor quality metrics** - Track accuracy, completeness, timeliness
4. **Alert on degradation** - Quality score < 95 = investigate
5. **Store rejected data** - Debug errors, improve validation
6. **Regular audits** - Manual review of edge cases
7. **Cross-vendor validation** - Compare IEX vs Polygon for conflicts

Now you can ensure data quality in production trading systems!
`,
};
