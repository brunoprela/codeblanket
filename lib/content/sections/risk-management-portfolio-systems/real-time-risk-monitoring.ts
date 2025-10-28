export const realTimeRiskMonitoring = `
# Real-Time Risk Monitoring

## Introduction

"In trading, yesterday's risk report is worthless. You need to know your risk RIGHT NOW."

Knight Capital lost $440M in 45 minutes (2012). By the time risk reports were generated, the firm was insolvent. Real-time risk monitoring is not optional - it's survival.

This section covers building production-grade, real-time risk surveillance systems that detect and respond to threats in milliseconds.

## Real-Time Risk Architecture

### 1. Streaming Risk Engine

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import asyncio
from enum import Enum

class RiskLevel(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"

@dataclass
class RiskMetric:
    """Single risk metric"""
    metric_id: str
    name: str
    current_value: float
    threshold_yellow: float
    threshold_orange: float
    threshold_red: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def risk_level(self) -> RiskLevel:
        """Determine risk level"""
        if self.current_value >= self.threshold_red:
            return RiskLevel.RED
        elif self.current_value >= self.threshold_orange:
            return RiskLevel.ORANGE
        elif self.current_value >= self.threshold_yellow:
            return RiskLevel.YELLOW
        else:
            return RiskLevel.GREEN

class RealTimeRiskMonitor:
    """
    Real-time risk monitoring system
    
    Features:
    - Streaming risk calculations
    - Sub-second latency
    - Automatic alerts
    - Circuit breakers
    """
    def __init__(self, max_history: int = 1000):
        self.metrics = {}
        self.history = deque(maxlen=max_history)
        self.alert_callbacks = []
        self.circuit_breakers = {}
        
    def register_metric(self, metric: RiskMetric):
        """Register risk metric for monitoring"""
        self.metrics[metric.metric_id] = metric
        
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)
        
    def update_metric(self, metric_id: str, new_value: float) -> Dict:
        """
        Update metric and check thresholds
        
        Returns alert info if threshold breached
        """
        if metric_id not in self.metrics:
            return {'error': f'Unknown metric: {metric_id}'}
        
        metric = self.metrics[metric_id]
        old_level = metric.risk_level
        old_value = metric.current_value
        
        # Update metric
        metric.current_value = new_value
        metric.timestamp = datetime.now()
        new_level = metric.risk_level
        
        # Store in history
        self.history.append({
            'timestamp': metric.timestamp,
            'metric_id': metric_id,
            'value': new_value,
            'risk_level': new_level.value
        })
        
        # Check if level changed
        if new_level != old_level:
            alert = self._create_alert(metric, old_level, new_level, old_value)
            self._trigger_alerts(alert)
            
            # Check circuit breakers
            if new_level == RiskLevel.RED:
                self._check_circuit_breakers(metric_id)
            
            return alert
        
        return {'status': 'ok', 'risk_level': new_level.value}
    
    def _create_alert(self, metric: RiskMetric, old_level: RiskLevel, 
                     new_level: RiskLevel, old_value: float) -> Dict:
        """Create alert message"""
        return {
            'type': 'RISK_LEVEL_CHANGE',
            'metric_id': metric.metric_id,
            'metric_name': metric.name,
            'old_level': old_level.value,
            'new_level': new_level.value,
            'old_value': old_value,
            'new_value': metric.current_value,
            'timestamp': metric.timestamp,
            'severity': self._get_severity(new_level)
        }
    
    def _get_severity(self, level: RiskLevel) -> str:
        """Map risk level to severity"""
        mapping = {
            RiskLevel.GREEN: 'INFO',
            RiskLevel.YELLOW: 'WARNING',
            RiskLevel.ORANGE: 'CRITICAL',
            RiskLevel.RED: 'EMERGENCY'
        }
        return mapping[level]
    
    def _trigger_alerts(self, alert: Dict):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def register_circuit_breaker(self, metric_id: str, action: Callable):
        """
        Register circuit breaker for metric
        
        Action is called when metric hits RED level
        """
        self.circuit_breakers[metric_id] = action
    
    def _check_circuit_breakers(self, metric_id: str):
        """Execute circuit breaker if registered"""
        if metric_id in self.circuit_breakers:
            print(f"🔴 CIRCUIT BREAKER TRIGGERED: {metric_id}")
            try:
                self.circuit_breakers[metric_id]()
            except Exception as e:
                print(f"Circuit breaker error: {e}")
    
    def get_dashboard_snapshot(self) -> Dict:
        """Get current risk dashboard snapshot"""
        snapshot = {
            'timestamp': datetime.now(),
            'metrics': {},
            'summary': {
                'green': 0,
                'yellow': 0,
                'orange': 0,
                'red': 0
            }
        }
        
        for metric_id, metric in self.metrics.items():
            level = metric.risk_level
            snapshot['metrics'][metric_id] = {
                'name': metric.name,
                'value': metric.current_value,
                'risk_level': level.value,
                'last_update': metric.timestamp
            }
            
            # Update summary
            level_key = level.value.lower()
            snapshot['summary'][level_key] += 1
        
        # Overall status
        if snapshot['summary']['red'] > 0:
            snapshot['overall_status'] = 'RED'
        elif snapshot['summary']['orange'] > 0:
            snapshot['overall_status'] = 'ORANGE'
        elif snapshot['summary']['yellow'] > 0:
            snapshot['overall_status'] = 'YELLOW'
        else:
            snapshot['overall_status'] = 'GREEN'
        
        return snapshot

# Alert Handler Example
def console_alert_handler(alert: Dict):
    """Print alerts to console"""
    severity = alert['severity']
    emoji = {
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'CRITICAL': '🚨',
        'EMERGENCY': '🔴'
    }
    
    print(f"{emoji.get(severity, '📊')} {severity}: {alert['metric_name']}")
    print(f"   {alert['old_level']} → {alert['new_level']}")
    print(f"   Value: {alert['old_value']:.2f} → {alert['new_value']:.2f}")
    print(f"   Time: {alert['timestamp']}")
    print()

# Example Usage
if __name__ == "__main__":
    # Create monitor
    monitor = RealTimeRiskMonitor()
    
    # Register alert handler
    monitor.register_alert_callback(console_alert_handler)
    
    # Define metrics
    var_metric = RiskMetric(
        metric_id='portfolio_var',
        name='Portfolio 99% VaR',
        current_value=2500000,
        threshold_yellow=4000000,
        threshold_orange=4500000,
        threshold_red=5000000
    )
    
    delta_metric = RiskMetric(
        metric_id='net_delta',
        name='Net Portfolio Delta',
        current_value=500000,
        threshold_yellow=800000,
        threshold_orange=900000,
        threshold_red=1000000
    )
    
    # Register metrics
    monitor.register_metric(var_metric)
    monitor.register_metric(delta_metric)
    
    print("Real-Time Risk Monitor")
    print("="*70)
    print()
    
    # Initial snapshot
    snapshot = monitor.get_dashboard_snapshot()
    print(f"Overall Status: {snapshot['overall_status']}")
    print(f"Metrics: {len(snapshot['metrics'])}")
    print()
    
    # Simulate risk increase
    print("Simulating market volatility spike...")
    print()
    monitor.update_metric('portfolio_var', 4100000)  # YELLOW
    monitor.update_metric('portfolio_var', 4600000)  # ORANGE
    monitor.update_metric('portfolio_var', 5200000)  # RED
    
    # Final snapshot
    snapshot = monitor.get_dashboard_snapshot()
    print(f"Final Status: {snapshot['overall_status']}")
\`\`\`

## Position-Level Real-Time Monitoring

\`\`\`python
class PositionMonitor:
    """
    Real-time position monitoring
    """
    def __init__(self):
        self.positions = {}
        self.prices = {}
        self.greeks = {}
        
    def update_position(self, symbol: str, quantity: float):
        """Update position"""
        self.positions[symbol] = quantity
        
    def update_price(self, symbol: str, price: float):
        """Update price and calculate risk"""
        old_price = self.prices.get(symbol, price)
        self.prices[symbol] = price
        
        # Calculate P&L impact
        if symbol in self.positions:
            quantity = self.positions[symbol]
            price_change = price - old_price
            pnl_impact = quantity * price_change
            
            return {
                'symbol': symbol,
                'quantity': quantity,
                'old_price': old_price,
                'new_price': price,
                'pnl_impact': pnl_impact,
                'position_value': quantity * price
            }
        
        return None
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate real-time portfolio metrics"""
        total_long = 0
        total_short = 0
        total_value = 0
        
        for symbol, quantity in self.positions.items():
            if symbol not in self.prices:
                continue
                
            value = quantity * self.prices[symbol]
            total_value += abs(value)
            
            if value > 0:
                total_long += value
            else:
                total_short += abs(value)
        
        net_exposure = total_long - total_short
        gross_exposure = total_long + total_short
        
        return {
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'long_exposure': total_long,
            'short_exposure': total_short,
            'num_positions': len(self.positions),
            'leverage': gross_exposure / net_exposure if net_exposure > 0 else 0
        }

# Example
if __name__ == "__main__":
    monitor = PositionMonitor()
    
    # Set positions
    monitor.update_position('AAPL', 10000)
    monitor.update_position('MSFT', -5000)
    monitor.update_position('GOOGL', 3000)
    
    # Set prices
    monitor.update_price('AAPL', 180.0)
    monitor.update_price('MSFT', 350.0)
    monitor.update_price('GOOGL', 140.0)
    
    print("Position Monitor")
    print("="*70)
    print()
    
    # Calculate metrics
    metrics = monitor.calculate_portfolio_metrics()
    print(f"Net Exposure: \${metrics['net_exposure']:,.0f}")
print(f"Gross Exposure: \${metrics['gross_exposure']:,.0f}")
print(f"Long: \${metrics['long_exposure']:,.0f}")
print(f"Short: \${metrics['short_exposure']:,.0f}")
print(f"Leverage: {metrics['leverage']:.2f}x")
print()
    
    # Price update
print("AAPL price drops to $175...")
result = monitor.update_price('AAPL', 175.0)
print(f"P&L Impact: \${result['pnl_impact']:,.0f}")
\`\`\`

## Kill Switch Implementation

\`\`\`python
class KillSwitch:
    """
    Emergency kill switch for trading
    
    Automatically stops all trading when triggered
    """
    def __init__(self, 
                 loss_threshold: float,
                 position_limit: float,
                 var_threshold: float):
        """
        Args:
            loss_threshold: Max acceptable loss
            position_limit: Max single position size
            var_threshold: Max VaR
        """
        self.loss_threshold = loss_threshold
        self.position_limit = position_limit
        self.var_threshold = var_threshold
        
        self.activated = False
        self.activation_reason = None
        self.activation_time = None
        
    def check_triggers(self, 
                      current_loss: float,
                      max_position: float,
                      current_var: float) -> Dict:
        """
        Check if kill switch should activate
        """
        if self.activated:
            return {
                'status': 'ALREADY_ACTIVATED',
                'reason': self.activation_reason,
                'time': self.activation_time
            }
        
        triggers = []
        
        # Loss trigger
        if current_loss < -self.loss_threshold:
            triggers.append({
                'type': 'LOSS_LIMIT',
                'current': current_loss,
                'threshold': -self.loss_threshold
            })
        
        # Position trigger
        if max_position > self.position_limit:
            triggers.append({
                'type': 'POSITION_LIMIT',
                'current': max_position,
                'threshold': self.position_limit
            })
        
        # VaR trigger
        if current_var > self.var_threshold:
            triggers.append({
                'type': 'VAR_LIMIT',
                'current': current_var,
                'threshold': self.var_threshold
            })
        
        # Activate if any triggers
        if triggers:
            self._activate(triggers)
            return {
                'status': 'ACTIVATED',
                'triggers': triggers,
                'time': self.activation_time
            }
        
        return {'status': 'OK'}
    
    def _activate(self, triggers: List[Dict]):
        """Activate kill switch"""
        self.activated = True
        self.activation_reason = triggers
        self.activation_time = datetime.now()
        
        print("=" * 70)
        print("🛑 KILL SWITCH ACTIVATED 🛑")
        print("=" * 70)
        print(f"Time: {self.activation_time}")
        print(f"Triggers: {len(triggers)}")
        for trigger in triggers:
            print(f"  - {trigger['type']}: {trigger['current']:.2f} > {trigger['threshold']:.2f}")
        print("=" * 70)
        print("ALL TRADING STOPPED")
        print("MANUAL OVERRIDE REQUIRED TO RESUME")
        print("=" * 70)
        
        # In production: close all positions, cancel all orders
        self._emergency_flatten()
    
    def _emergency_flatten(self):
        """Emergency position flattening"""
        print("Executing emergency position flatten...")
        # In production:
        # 1. Cancel all pending orders
        # 2. Close all positions at market
        # 3. Disconnect from trading venues
        # 4. Alert risk management
        print("All positions closed.")
    
    def manual_reset(self, authorized_user: str):
        """Manual reset by authorized user"""
        if not self.activated:
            return {'status': 'NOT_ACTIVATED'}
        
        print(f"Kill switch reset by: {authorized_user}")
        print(f"Previous activation: {self.activation_time}")
        print(f"Reason: {self.activation_reason}")
        
        self.activated = False
        self.activation_reason = None
        self.activation_time = None
        
        return {'status': 'RESET', 'user': authorized_user}

# Example
if __name__ == "__main__":
    kill_switch = KillSwitch(
        loss_threshold=500000,    # $500K
        position_limit=100000000, # $100M
        var_threshold=5000000     # $5M
    )
    
    print("Kill Switch System")
    print("="*70)
    print()
    
    # Normal operation
    result = kill_switch.check_triggers(
        current_loss=-200000,
        max_position=50000000,
        current_var=3000000
    )
    print(f"Status: {result['status']}")
    print()
    
    # Trigger event
    print("Catastrophic loss detected...")
    result = kill_switch.check_triggers(
        current_loss=-650000,  # Exceeds threshold
        max_position=50000000,
        current_var=3000000
    )
\`\`\`

## Real-World: Knight Capital (2012)

**What Happened**:
- Software deployment error activated old trading code
- Bought and sold 397 million shares in 45 minutes
- $440M loss before kill switch activated
- Firm insolvent

**Real-Time Monitoring Failures**:
1. **No anomaly detection**: 397M shares unusual not detected
2. **Delayed response**: 45 minutes to stop trading
3. **No position limits**: Accumulated massive positions
4. **Manual kill switch**: Should have been automatic

**Lessons**:
- Automated anomaly detection required
- Kill switches must be automatic
- Real-time position monitoring essential
- Pre-trade risk checks critical

## Anomaly Detection

\`\`\`python
class AnomalyDetector:
    """
    Statistical anomaly detection for risk metrics
    """
    def __init__(self, window_size: int = 100, threshold_std: float = 3.0):
        """
        Args:
            window_size: Rolling window for statistics
            threshold_std: Standard deviations for alert
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.history = deque(maxlen=window_size)
        
    def update(self, value: float) -> Dict:
        """
        Update with new value and check for anomaly
        """
        self.history.append(value)
        
        if len(self.history) < 10:  # Need minimum data
            return {'anomaly': False, 'reason': 'insufficient_data'}
        
        # Calculate statistics
        mean = np.mean(self.history)
        std = np.std(self.history)
        
        # Z-score
        z_score = (value - mean) / std if std > 0 else 0
        
        # Check if anomaly
        is_anomaly = abs(z_score) > self.threshold_std
        
        return {
            'anomaly': is_anomaly,
            'value': value,
            'mean': mean,
            'std': std,
            'z_score': z_score,
            'threshold': self.threshold_std
        }

# Example
detector = AnomalyDetector(window_size=100, threshold_std=3.0)

# Normal values
for i in range(50):
    value = np.random.normal(100, 10)
    result = detector.update(value)

# Anomaly
anomaly_value = 200
result = detector.update(anomaly_value)

if result['anomaly']:
    print(f"🚨 ANOMALY DETECTED")
    print(f"   Value: {result['value']:.2f}")
    print(f"   Mean: {result['mean']:.2f}")
    print(f"   Z-Score: {result['z_score']:.2f}")
\`\`\`

## Key Takeaways

1. **Sub-Second Latency**: Risk updates must be real-time
2. **Automated Alerts**: Human cannot react fast enough
3. **Circuit Breakers**: Automatic stops at thresholds
4. **Kill Switch**: Ultimate emergency stop
5. **Anomaly Detection**: Statistical detection of unusual activity
6. **Multiple Layers**: Position, P&L, VaR, concentration
7. **Independent System**: Cannot be disabled by traders

## Production Checklist

- [ ] Real-time risk calculations (<100ms latency)
- [ ] Automated alert system
- [ ] Circuit breakers for each metric
- [ ] Emergency kill switch
- [ ] Anomaly detection
- [ ] Dashboard with color-coded status
- [ ] Historical tracking
- [ ] Independent risk infrastructure
- [ ] Tested failover procedures
- [ ] Regular fire drills

## Conclusion

Real-time risk monitoring is the early warning system that prevents catastrophes. Knight Capital shows what happens without it: complete failure in 45 minutes.

Modern risk systems must detect and respond to threats faster than humans can. Automation is not optional - it's survival.

Next: Risk Reporting & Dashboards - communicating risk to stakeholders.
`;

