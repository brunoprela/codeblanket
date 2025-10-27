export const systemMonitoringAlerting = {
  title: 'System Monitoring and Alerting',
  id: 'system-monitoring-alerting',
  content: `
# System Monitoring and Alerting

## Introduction

**System monitoring** is the nervous system of trading infrastructure - detecting issues before they impact P&L. In trading, every second of downtime or degraded performance can cost millions.

**Why Monitoring is Critical:**
- **Financial impact**: Knight Capital lost $440M in 45 minutes due to unmonitored software bug (2012)
- **Regulatory compliance**: SEC requires systems monitoring and incident response plans
- **Client trust**: Institutional clients demand 99.99% uptime SLAs
- **Risk management**: Real-time alerts on position/P&L limit breaches
- **Performance**: Detect latency spikes before they cause missed fills

**Real-World Monitoring Stacks:**
- **Citadel Securities**: Custom metrics platform + Prometheus + Grafana, sub-second alerting
- **Jane Street**: OCaml-based monitoring, real-time P&L dashboards updated every 100ms
- **Interactive Brokers**: Multi-datacenter monitoring, 24/7 NOC, <30s incident detection
- **Bloomberg**: Proprietary monitoring (B-Unit), monitors 325,000+ functions globally

**Modern Monitoring Architecture:**

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    MONITORING ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Trading System Components                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │   OMS    │ │   EMS    │ │   Risk   │ │ Position │       │
│  │          │ │          │ │          │ │  Tracker │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │            │            │            │               │
│       └────────────┴────────────┴────────────┘               │
│                    │                                          │
│                    ▼                                          │
│       ┌──────────────────────────┐                          │
│       │    Metrics Collection     │                          │
│       │  (Prometheus Exporter)    │                          │
│       └──────────┬───────────────┘                          │
│                  │                                            │
│                  ▼                                            │
│       ┌──────────────────────────┐                          │
│       │      Prometheus           │                          │
│       │   (Time-Series DB)        │                          │
│       └──────────┬───────────────┘                          │
│                  │                                            │
│       ┌──────────┴───────────┐                              │
│       │                      │                               │
│       ▼                      ▼                               │
│  ┌─────────┐         ┌──────────────┐                      │
│  │ Grafana │         │ Alert Manager│                       │
│  │Dashboard│         │              │                       │
│  └─────────┘         └──────┬───────┘                      │
│                              │                               │
│                              ▼                               │
│                     ┌────────────────┐                      │
│                     │   PagerDuty    │                      │
│                     │   Slack/Email  │                      │
│                     └────────────────┘                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
\`\`\`

This section covers production monitoring and alerting for trading systems.

---

## Prometheus Metrics Collection

\`\`\`python
"""
Comprehensive Prometheus Metrics for Trading Systems
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, push_to_gateway,
    start_http_server, generate_latest
)
from typing import Dict, List
import time
from datetime import datetime
from decimal import Decimal
from functools import wraps

class TradingMetrics:
    """
    Production metrics for trading system
    
    Metric types:
    - Counter: Monotonically increasing (total orders, fills)
    - Gauge: Can go up/down (positions, P&L, latency)
    - Histogram: Distributions (latency percentiles)
    - Summary: Similar to histogram but client-side
    """
    
    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()
        
        # ============================================================
        # ORDER METRICS
        # ============================================================
        
        # Total orders by status
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders',
            ['symbol', 'side', 'order_type', 'status'],
            registry=self.registry
        )
        
        # Order rejection rate
        self.orders_rejected = Counter(
            'trading_orders_rejected_total',
            'Orders rejected',
            ['symbol', 'reason'],
            registry=self.registry
        )
        
        # Order success rate (derived metric)
        self.orders_accepted = Counter(
            'trading_orders_accepted_total',
            'Orders accepted',
            ['symbol'],
            registry=self.registry
        )
        
        # Order fill rate
        self.orders_filled = Counter(
            'trading_orders_filled_total',
            'Orders filled',
            ['symbol', 'venue'],
            registry=self.registry
        )
        
        # Order latency (end-to-end)
        self.order_latency = Histogram(
            'trading_order_latency_seconds',
            'Order processing latency',
            ['order_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        # Order volume (notional)
        self.order_volume = Counter(
            'trading_order_volume_usd',
            'Order volume in USD',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        # ============================================================
        # POSITION METRICS
        # ============================================================
        
        # Current position by symbol
        self.position_quantity = Gauge(
            'trading_position_quantity',
            'Current position quantity',
            ['symbol', 'account'],
            registry=self.registry
        )
        
        # Position value (market value)
        self.position_value = Gauge(
            'trading_position_value_usd',
            'Position market value in USD',
            ['symbol', 'account'],
            registry=self.registry
        )
        
        # Position concentration (% of portfolio)
        self.position_concentration = Gauge(
            'trading_position_concentration_pct',
            'Position as % of portfolio',
            ['symbol'],
            registry=self.registry
        )
        
        # ============================================================
        # P&L METRICS
        # ============================================================
        
        # Realized P&L
        self.realized_pnl = Gauge(
            'trading_realized_pnl_usd',
            'Realized P&L in USD',
            ['symbol', 'strategy'],
            registry=self.registry
        )
        
        # Unrealized P&L
        self.unrealized_pnl = Gauge(
            'trading_unrealized_pnl_usd',
            'Unrealized P&L in USD',
            ['symbol', 'strategy'],
            registry=self.registry
        )
        
        # Total portfolio P&L
        self.portfolio_pnl = Gauge(
            'trading_portfolio_pnl_usd',
            'Total portfolio P&L',
            ['account'],
            registry=self.registry
        )
        
        # Daily P&L high water mark
        self.pnl_high_water_mark = Gauge(
            'trading_pnl_high_water_mark_usd',
            'Daily P&L high water mark',
            registry=self.registry
        )
        
        # ============================================================
        # EXECUTION METRICS
        # ============================================================
        
        # Fill latency (order → fill)
        self.fill_latency = Histogram(
            'trading_fill_latency_seconds',
            'Time from order to fill',
            ['venue'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Slippage (execution price vs expected)
        self.slippage = Histogram(
            'trading_slippage_bps',
            'Slippage in basis points',
            ['symbol', 'side'],
            buckets=[-10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50],
            registry=self.registry
        )
        
        # Commission costs
        self.commission = Counter(
            'trading_commission_usd',
            'Commission paid in USD',
            ['venue', 'symbol'],
            registry=self.registry
        )
        
        # ============================================================
        # RISK METRICS
        # ============================================================
        
        # Position limit utilization
        self.position_limit_utilization = Gauge(
            'trading_position_limit_utilization_pct',
            'Position limit utilization %',
            ['symbol'],
            registry=self.registry
        )
        
        # VaR (Value at Risk)
        self.var = Gauge(
            'trading_var_usd',
            'Value at Risk (95%)',
            ['portfolio'],
            registry=self.registry
        )
        
        # Risk limit breaches
        self.risk_breaches = Counter(
            'trading_risk_breaches_total',
            'Risk limit breaches',
            ['limit_type', 'severity'],
            registry=self.registry
        )
        
        # ============================================================
        # SYSTEM HEALTH METRICS
        # ============================================================
        
        # Component health (0=down, 1=up)
        self.component_health = Gauge(
            'trading_component_health',
            'Component health status',
            ['component'],
            registry=self.registry
        )
        
        # Market data lag
        self.market_data_lag = Gauge(
            'trading_market_data_lag_ms',
            'Market data latency in ms',
            ['exchange', 'symbol'],
            registry=self.registry
        )
        
        # Database connection pool
        self.db_connections = Gauge(
            'trading_db_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        
        # Memory usage
        self.memory_usage = Gauge(
            'trading_memory_usage_mb',
            'Memory usage in MB',
            ['component'],
            registry=self.registry
        )
        
        # ============================================================
        # RECONCILIATION METRICS
        # ============================================================
        
        # Trade breaks
        self.trade_breaks = Counter(
            'trading_trade_breaks_total',
            'Trade reconciliation breaks',
            ['break_type'],
            registry=self.registry
        )
        
        # Reconciliation rate
        self.reconciliation_rate = Gauge(
            'trading_reconciliation_rate_pct',
            'Percentage of trades reconciled',
            registry=self.registry
        )
    
    def record_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        quantity: Decimal,
        price: Decimal,
        latency_seconds: float
    ):
        """Record order metrics"""
        # Count order
        self.orders_total.labels(
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=status
        ).inc()
        
        # Record latency
        self.order_latency.labels(order_type=order_type).observe(latency_seconds)
        
        # Record volume
        notional = float(quantity * price)
        self.order_volume.labels(symbol=symbol, side=side).inc(notional)
        
        # Track acceptance
        if status == 'ACCEPTED':
            self.orders_accepted.labels(symbol=symbol).inc()
        elif status == 'REJECTED':
            self.orders_rejected.labels(symbol=symbol, reason='validation').inc()
    
    def record_position(
        self,
        symbol: str,
        account: str,
        quantity: Decimal,
        market_value: Decimal,
        portfolio_value: Decimal
    ):
        """Record position metrics"""
        self.position_quantity.labels(symbol=symbol, account=account).set(float(quantity))
        self.position_value.labels(symbol=symbol, account=account).set(float(market_value))
        
        # Position concentration
        concentration = float(market_value / portfolio_value * 100) if portfolio_value > 0 else 0
        self.position_concentration.labels(symbol=symbol).set(concentration)
    
    def record_pnl(
        self,
        symbol: str,
        strategy: str,
        account: str,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal
    ):
        """Record P&L metrics"""
        self.realized_pnl.labels(symbol=symbol, strategy=strategy).set(float(realized_pnl))
        self.unrealized_pnl.labels(symbol=symbol, strategy=strategy).set(float(unrealized_pnl))
        
        # Total P&L
        total_pnl = float(realized_pnl + unrealized_pnl)
        self.portfolio_pnl.labels(account=account).set(total_pnl)
    
    def record_fill(
        self,
        symbol: str,
        venue: str,
        side: str,
        expected_price: Decimal,
        actual_price: Decimal,
        commission: Decimal,
        latency_seconds: float
    ):
        """Record fill metrics"""
        # Fill latency
        self.fill_latency.labels(venue=venue).observe(latency_seconds)
        
        # Slippage (in basis points)
        slippage_bps = float((actual_price - expected_price) / expected_price * 10000)
        if side == 'SELL':
            slippage_bps = -slippage_bps
        self.slippage.labels(symbol=symbol, side=side).observe(slippage_bps)
        
        # Commission
        self.commission.labels(venue=venue, symbol=symbol).inc(float(commission))
    
    def set_component_health(self, component: str, is_healthy: bool):
        """Set component health"""
        self.component_health.labels(component=component).set(1.0 if is_healthy else 0.0)


def measure_latency(metrics: TradingMetrics, operation: str):
    """Decorator to measure operation latency"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.perf_counter() - start
                # Record latency based on operation
                if 'order' in operation.lower():
                    metrics.order_latency.labels(order_type='MARKET').observe(latency)
        return wrapper
    return decorator


# Example usage
metrics = TradingMetrics()

# Start metrics HTTP server (Prometheus scrapes this)
start_http_server(8000)

# Record order
metrics.record_order(
    symbol='AAPL',
    side='BUY',
    order_type='MARKET',
    status='ACCEPTED',
    quantity=Decimal('100'),
    price=Decimal('150.00'),
    latency_seconds=0.025
)

# Record position
metrics.record_position(
    symbol='AAPL',
    account='ACC-001',
    quantity=Decimal('100'),
    market_value=Decimal('15000'),
    portfolio_value=Decimal('100000')
)

# Record P&L
metrics.record_pnl(
    symbol='AAPL',
    strategy='momentum',
    account='ACC-001',
    realized_pnl=Decimal('250.00'),
    unrealized_pnl=Decimal('500.00')
)
\`\`\`

---

## Grafana Dashboard Configuration

\`\`\`json
{
  "dashboard": {
    "title": "Trading System Dashboard",
    "tags": ["trading", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Orders per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(trading_orders_total[1m])",
            "legendFormat": "{{symbol}} {{side}}",
            "refId": "A"
          }
        ],
        "yaxes": [
          {"label": "Orders/sec", "format": "short"}
        ],
        "alert": {
          "conditions": [
            {
              "evaluator": {"params": [1000], "type": "gt"},
              "operator": {"type": "and"},
              "query": {"params": ["A", "5m", "now"]},
              "reducer": {"params": [], "type": "avg"},
              "type": "query"
            }
          ],
          "executionErrorState": "alerting",
          "frequency": "60s",
          "handler": 1,
          "name": "High Order Rate",
          "notifications": []
        }
      },
      {
        "id": 2,
        "title": "Order Latency (p50, p95, p99)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(trading_order_latency_seconds_bucket[5m]))",
            "legendFormat": "p50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, rate(trading_order_latency_seconds_bucket[5m]))",
            "legendFormat": "p95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, rate(trading_order_latency_seconds_bucket[5m]))",
            "legendFormat": "p99",
            "refId": "C"
          }
        ],
        "yaxes": [
          {"label": "Latency (seconds)", "format": "s"}
        ],
        "thresholds": [
          {"value": 0.1, "colorMode": "critical", "fill": true, "line": true, "op": "gt"}
        ]
      },
      {
        "id": 3,
        "title": "Portfolio P&L",
        "type": "gauge",
        "targets": [
          {
            "expr": "trading_portfolio_pnl_usd",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": -50000, "color": "red"},
                {"value": -10000, "color": "orange"},
                {"value": 0, "color": "yellow"},
                {"value": 10000, "color": "green"}
              ]
            },
            "unit": "currencyUSD"
          }
        }
      },
      {
        "id": 4,
        "title": "Position Concentration",
        "type": "piechart",
        "targets": [
          {
            "expr": "trading_position_value_usd",
            "legendFormat": "{{symbol}}",
            "refId": "A"
          }
        ]
      },
      {
        "id": 5,
        "title": "Order Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(trading_orders_accepted_total[5m])) / sum(rate(trading_orders_total[5m])) * 100",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"value": 0, "color": "red"},
                {"value": 95, "color": "yellow"},
                {"value": 99, "color": "green"}
              ]
            },
            "unit": "percent"
          }
        }
      },
      {
        "id": 6,
        "title": "Fill Latency by Venue",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(trading_fill_latency_seconds_bucket[5m])",
            "legendFormat": "{{venue}}",
            "refId": "A"
          }
        ]
      },
      {
        "id": 7,
        "title": "Risk Limit Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "trading_position_limit_utilization_pct",
            "legendFormat": "{{symbol}}",
            "refId": "A"
          }
        ],
        "thresholds": [
          {"value": 80, "colorMode": "warning", "op": "gt"},
          {"value": 95, "colorMode": "critical", "op": "gt"}
        ]
      },
      {
        "id": 8,
        "title": "System Component Health",
        "type": "table",
        "targets": [
          {
            "expr": "trading_component_health",
            "format": "table",
            "instant": true,
            "refId": "A"
          }
        ],
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {},
              "indexByName": {"component": 0, "Value": 1},
              "renameByName": {"component": "Component", "Value": "Status"}
            }
          }
        ]
      }
    ],
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"}
  }
}
\`\`\`

---

## Alerting Rules

\`\`\`yaml
# Prometheus Alert Manager Configuration

groups:
  - name: trading_critical_alerts
    interval: 10s
    rules:
      # ============================================================
      # LATENCY ALERTS
      # ============================================================
      
      - alert: HighOrderLatency
        expr: histogram_quantile(0.99, rate(trading_order_latency_seconds_bucket[5m])) > 0.1
        for: 2m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High order latency detected"
          description: "p99 order latency is {{ $value }}s (threshold: 100ms)"
          runbook_url: "https://wiki.company.com/runbooks/high-latency"
      
      - alert: CriticalOrderLatency
        expr: histogram_quantile(0.99, rate(trading_order_latency_seconds_bucket[5m])) > 0.5
        for: 1m
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "CRITICAL: Order latency exceeded 500ms"
          description: "p99 latency: {{ $value }}s - Immediate action required"
          runbook_url: "https://wiki.company.com/runbooks/critical-latency"
      
      # ============================================================
      # P&L ALERTS
      # ============================================================
      
      - alert: DailyLossLimitApproaching
        expr: trading_portfolio_pnl_usd < -40000
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Daily loss approaching limit"
          description: "Portfolio P&L: ${{ $value }} (limit: -$50K)"
      
      - alert: DailyLossLimitBreached
        expr: trading_portfolio_pnl_usd < -50000
        for: 30s
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "CRITICAL: Daily loss limit breached"
          description: "Portfolio P&L: ${{ $value }} - STOP TRADING"
          runbook_url: "https://wiki.company.com/runbooks/loss-limit-breach"
      
      - alert: LargePnLMove
        expr: abs(delta(trading_portfolio_pnl_usd[5m])) > 10000
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Large P&L move detected"
          description: "P&L moved ${{ $value }} in 5 minutes"
      
      # ============================================================
      # ORDER ALERTS
      # ============================================================
      
      - alert: HighOrderRejectionRate
        expr: rate(trading_orders_rejected_total[5m]) / rate(trading_orders_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High order rejection rate"
          description: "{{ $value | humanizePercentage }} of orders rejected"
      
      - alert: NoOrdersReceived
        expr: rate(trading_orders_total[5m]) == 0
        for: 5m
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "No orders received"
          description: "No orders in last 5 minutes - System down?"
      
      - alert: OrderSuccessRateLow
        expr: sum(rate(trading_orders_filled_total[5m])) / sum(rate(trading_orders_accepted_total[5m])) < 0.95
        for: 5m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Order fill rate below 95%"
          description: "Only {{ $value | humanizePercentage }} of orders filling"
      
      # ============================================================
      # POSITION ALERTS
      # ============================================================
      
      - alert: PositionLimitApproaching
        expr: trading_position_limit_utilization_pct > 80
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Position limit utilization high"
          description: "{{$labels.symbol}} position at {{ $value }}% of limit"
      
      - alert: PositionLimitBreached
        expr: trading_position_limit_utilization_pct > 100
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "CRITICAL: Position limit breached"
          description: "{{$labels.symbol}} exceeded position limit"
      
      - alert: ConcentrationRisk
        expr: trading_position_concentration_pct > 25
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "High position concentration"
          description: "{{$labels.symbol}} is {{ $value }}% of portfolio"
      
      # ============================================================
      # SYSTEM HEALTH ALERTS
      # ============================================================
      
      - alert: ComponentDown
        expr: trading_component_health == 0
        for: 30s
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "CRITICAL: Component {{$labels.component}} is DOWN"
          description: "Component health check failed"
          runbook_url: "https://wiki.company.com/runbooks/component-down"
      
      - alert: HighMarketDataLag
        expr: trading_market_data_lag_ms > 1000
        for: 1m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Market data lag high"
          description: "{{$labels.exchange}} {{$labels.symbol}}: {{ $value }}ms lag"
      
      - alert: DatabaseConnectionPoolExhausted
        expr: trading_db_connections_active > 90
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Database connection pool nearly full"
          description: "{{ $value }} active connections (max: 100)"
      
      # ============================================================
      # RECONCILIATION ALERTS
      # ============================================================
      
      - alert: HighTradeBreakRate
        expr: rate(trading_trade_breaks_total[15m]) > 10
        labels:
          severity: warning
          team: operations
        annotations:
          summary: "High trade reconciliation break rate"
          description: "{{ $value }} breaks per minute"
      
      - alert: ReconciliationRateLow
        expr: trading_reconciliation_rate_pct < 99
        for: 10m
        labels:
          severity: critical
          team: operations
        annotations:
          summary: "Reconciliation rate below 99%"
          description: "Only {{ $value }}% of trades reconciled"

# Alert Manager routing
route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 30s
  repeat_interval: 4h
  
  routes:
    # Critical alerts → PagerDuty (immediate)
    - match:
        severity: critical
      receiver: pagerduty
      continue: true
    
    # Warning alerts → Slack
    - match:
        severity: warning
      receiver: slack
    
    # After-hours critical → Escalate
    - match:
        severity: critical
      receiver: pagerduty_escalation
      active_time_intervals:
        - after_hours

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://alertmanager-webhook:8080/alerts'
  
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}'
        details:
          firing: '{{ template "pagerduty.default.description" . }}'
  
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#trading-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true
  
  - name: 'pagerduty_escalation'
    pagerduty_configs:
      - service_key: 'YOUR_ESCALATION_KEY'
        severity: 'critical'

time_intervals:
  - name: after_hours
    time_intervals:
      - times:
          - start_time: '18:00'
            end_time: '09:00'
        weekdays: ['monday:friday']
      - weekdays: ['saturday', 'sunday']
\`\`\`

---

## Log Aggregation

\`\`\`python
"""
Structured Logging for Trading Systems
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any
import sys

class StructuredLogger:
    """
    JSON structured logging for aggregation
    
    Sends logs to:
    - Elasticsearch (via Filebeat)
    - Splunk
    - Datadog
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def _log(self, level: str, message: str, **kwargs):
        """Log structured message"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name,
            'level': level,
            'message': message,
            **kwargs
        }
        
        self.logger.log(
            getattr(logging, level),
            json.dumps(log_entry)
        )
    
    def info(self, message: str, **kwargs):
        """Info log"""
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning log"""
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error log"""
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical log"""
        self._log('CRITICAL', message, **kwargs)
    
    # Trading-specific log methods
    
    def log_order(self, order_id: str, symbol: str, side: str, quantity: float, status: str):
        """Log order event"""
        self.info(
            f"Order {order_id}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=status,
            event_type='order'
        )
    
    def log_fill(self, order_id: str, fill_id: str, quantity: float, price: float):
        """Log fill event"""
        self.info(
            f"Fill {fill_id}",
            order_id=order_id,
            fill_id=fill_id,
            quantity=quantity,
            price=price,
            event_type='fill'
        )
    
    def log_risk_breach(self, symbol: str, limit_type: str, limit: float, actual: float):
        """Log risk breach"""
        self.critical(
            f"Risk breach: {limit_type}",
            symbol=symbol,
            limit_type=limit_type,
            limit=limit,
            actual=actual,
            breach_pct=(actual / limit - 1) * 100,
            event_type='risk_breach'
        )


# Example usage
logger = StructuredLogger('trading-oms')

logger.log_order(
    order_id='ORD-12345',
    symbol='AAPL',
    side='BUY',
    quantity=100,
    status='FILLED'
)

logger.log_risk_breach(
    symbol='TSLA',
    limit_type='position_limit',
    limit=10000,
    actual=12000
)

# Output (JSON for parsing by log aggregators):
# {"timestamp": "2025-10-26T10:30:00", "service": "trading-oms", ...}
\`\`\`

---

## Summary

**Monitoring Stack:**

| Component | Purpose | Metrics | Alerts |
|-----------|---------|---------|--------|
| Prometheus | Metrics storage | Orders, P&L, Latency | Alert rules |
| Grafana | Visualization | Real-time dashboards | Threshold alerts |
| Alert Manager | Alert routing | Deduplication | PagerDuty, Slack |
| ELK Stack | Log aggregation | Error logs, Audit trail | Log-based alerts |
| Datadog/New Relic | APM | Distributed tracing | Performance alerts |

**Key Metrics to Monitor:**1. **Latency**: Order latency (p50/p95/p99 <100ms target)
2. **Throughput**: Orders/sec, Fills/sec
3. **Success Rate**: Order acceptance rate (>99%), Fill rate (>95%)
4. **P&L**: Real-time P&L, Daily high/low, Limit utilization
5. **System Health**: Component uptime, Database connections, Memory usage
6. **Risk**: Position concentration, VaR, Limit utilization

**Alert Severity Levels:**
- **Critical** (PagerDuty): System down, P&L limit breach, Component failure
- **Warning** (Slack): High latency, Order rejection rate >5%, Position approaching limit
- **Info** (Logs): Normal operations, Successful orders

**Production Best Practices:**1. Monitor everything that can impact P&L
2. Alert on trends, not just absolute values
3. Set alert thresholds with 20% buffer before actual limits
4. Test alerts monthly (chaos engineering)
5. Document runbooks for each alert
6. Review alert fatigue weekly (disable noisy alerts)
7. Require < 2 minute response time for critical alerts

**Next Section**: Module 14.13 - Disaster Recovery and Failover
`,
};
