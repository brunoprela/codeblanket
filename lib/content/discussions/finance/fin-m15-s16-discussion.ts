export default {
  id: 'fin-m15-s16-discussion',
  title: 'Risk Management Platform Project - Discussion Questions',
  questions: [
    {
      question:
        'Design the architecture for a risk management platform that handles real-time position tracking, VaR calculation, limit monitoring, and reporting for a multi-desk trading organization. What are the key components, data flows, and technology choices?',
      answer: `A production risk management platform requires careful architectural design:

**System Architecture Overview**
\`\`\`python
risk_platform_architecture = {
    'Layer_1_Data_Ingestion': {
        'sources': [
            'Trading systems (OMS/EMS)',
            'Market data (prices, vols)',
            'Reference data (securities master)',
            'Risk-free rates, FX rates'
        ],
        'technology': 'Kafka, Solace (message queues)',
        'latency': '<100ms',
        'throughput': '100K messages/second'
    },
    
    'Layer_2_Position_Management': {
        'database': 'Redis (in-memory) + PostgreSQL (persistent)',
        'structure': 'Real-time position cache',
        'updates': 'Every trade, corporate action',
        'access_pattern': 'Read-heavy (10000:1 read:write)',
        'latency': '<10ms for position lookup'
    },
    
    'Layer_3_Risk_Calculation': {
        'engines': [
            'VaR engine (C++ for speed)',
            'Stress test engine (Python/distributed)',
            'Attribution engine',
            'Limit checker'
        ],
        'compute': 'GPU cluster for Monte Carlo',
        'parallelization': 'Spark for distributed calcs',
        'caching': 'Cache Greeks, reuse when possible'
    },
    
    'Layer_4_Limit_Monitoring': {
        'real_time': 'Pre-trade and post-trade checks',
        'database': 'Redis (limits, current usage)',
        'alerts': 'Kafka → alert service',
        'latency': '<50ms for limit check'
    },
    
    'Layer_5_Reporting': {
        'batch': 'EOD regulatory reports',
        'real_time': 'Dashboard (WebSocket)',
        'storage': 'Data warehouse (Snowflake, BigQuery)',
        'visualization': 'React frontend + D3.js'
    },
    
    'Layer_6_APIs': {
        'rest_api': 'External integrations',
        'websocket': 'Real-time streaming',
        'graphql': 'Flexible data queries',
        'authentication': 'OAuth 2.0 + JWT'
    }
}
\`\`\`

**Data Flow Architecture**
\`\`\`python
# Real-time trade processing

data_flow = {
    'Trade_Executed': {
        '1_oms': 'Trade executed in OMS',
        '2_kafka': 'Trade message → Kafka topic',
        '3_consumer': 'Risk system consumes message',
        '4_position_update': 'Update position cache (Redis)',
        '5_risk_calc': 'Trigger incremental VaR update',
        '6_limit_check': 'Check if VaR exceeds limit',
        '7_alert': 'If breach, send alert (Kafka)',
        '8_dashboard': 'Update dashboard (WebSocket)',
        '9_persistence': 'Persist to PostgreSQL (async)',
        
        'total_latency': '<500ms end-to-end'
    },
    
    'EOD_Processing': {
        '1_snapshot': 'Take position snapshot',
        '2_full_var': 'Calculate full Monte Carlo VaR',
        '3_stress_tests': 'Run all stress scenarios',
        '4_attribution': 'P&L and risk attribution',
        '5_reports': 'Generate regulatory reports',
        '6_warehouse': 'Load to data warehouse',
        '7_archive': 'Archive to S3',
        
        'total_time': '30-60 minutes'
    }
}
\`\`\`

**Technology Stack**
\`\`\`python
tech_stack = {
    'Backend': {
        'risk_engine': 'C++ (performance critical)',
        'api_services': 'Python (FastAPI)',
        'batch_processing': 'Python (Pandas, NumPy)',
        'distributed_compute': 'PySpark'
    },
    
    'Data_Infrastructure': {
        'message_queue': 'Kafka',
        'cache': 'Redis',
        'oltp': 'PostgreSQL',
        'olap': 'Snowflake / BigQuery',
        'object_storage': 'S3',
        'time_series': 'TimescaleDB'
    },
    
    'Compute': {
        'risk_calcs': 'CUDA (GPU)',
        'orchestration': 'Kubernetes',
        'workflow': 'Airflow',
        'serverless': 'AWS Lambda (light tasks)'
    },
    
    'Frontend': {
        'framework': 'React + TypeScript',
        'state': 'Redux',
        'charts': 'D3.js, Recharts',
        'real_time': 'WebSocket',
        'mobile': 'React Native'
    },
    
    'DevOps': {
        'container': 'Docker',
        'orchestration': 'Kubernetes',
        'ci_cd': 'GitHub Actions',
        'monitoring': 'Datadog, Grafana',
        'logging': 'ELK stack'
    }
}
\`\`\`

**Database Schema Design**
\`\`\`python
# Core tables

database_schema = {
    'positions': {
        'columns': [
            'position_id (PK)',
            'account_id',
            'security_id',
            'quantity',
            'average_cost',
            'current_price',
            'market_value',
            'updated_at'
        ],
        'indexes': ['account_id', 'security_id', 'updated_at'],
        'partitioning': 'By date',
        'volume': '10M+ rows'
    },
    
    'risk_metrics': {
        'columns': [
            'metric_id (PK)',
            'account_id',
            'desk_id',
            'metric_type (VaR, stress, etc.)',
            'value',
            'timestamp',
            'calculation_method'
        ],
        'indexes': ['account_id', 'timestamp', 'metric_type'],
        'partitioning': 'By date',
        'retention': '3 years online, 10 years archived'
    },
    
    'limits': {
        'columns': [
            'limit_id (PK)',
            'entity_id (trader/desk/firm)',
            'limit_type',
            'limit_value',
            'effective_date',
            'expiry_date'
        ],
        'indexes': ['entity_id', 'limit_type'],
        'volume': '10K rows'
    },
    
    'alerts': {
        'columns': [
            'alert_id (PK)',
            'entity_id',
            'alert_type',
            'severity',
            'message',
            'timestamp',
            'acknowledged',
            'resolved'
        ],
        'indexes': ['entity_id', 'timestamp', 'acknowledged'],
        'retention': '90 days online, archive after'
    }
}
\`\`\`

**Key Design Decisions**

**1. Real-Time vs Batch**
\`\`\`python
real_time_vs_batch = {
    'Real_Time': {
        'use_for': [
            'Pre-trade limit checks',
            'Post-trade position updates',
            'Intraday P&L',
            'Incremental VaR',
            'Dashboard updates'
        ],
        'technology': 'In-memory (Redis), event-driven (Kafka)',
        'priority': 'Latency < accuracy',
        'approximations': 'OK (incremental VaR 95% accurate)'
    },
    
    'Batch_EOD': {
        'use_for': [
            'Full Monte Carlo VaR',
            'Comprehensive stress testing',
            'Regulatory reports',
            'Historical analysis'
        ],
        'technology': 'Data warehouse, distributed compute',
        'priority': 'Accuracy > speed',
        'approximations': 'Not acceptable (must be exact)'
    }
}

# Hybrid approach: Real-time for ops, batch for accuracy
\`\`\`

**2. Scaling Strategy**
\`\`\`python
scaling_approach = {
    'Horizontal_Scaling': {
        'components': [
            'API servers (stateless)',
            'Risk calculation workers',
            'Kafka consumers'
        ],
        'method': 'Add more instances',
        'orchestration': 'Kubernetes auto-scaling',
        'cost': 'Linear with load'
    },
    
    'Caching_Strategy': {
        'what_to_cache': [
            'Positions (Redis)',
            'Market data (last 1 hour)',
            'Greeks/sensitivities',
            'Correlation matrices (daily update)'
        ],
        'ttl': 'Position: 1s, Market: 5s, Greeks: 1hr',
        'invalidation': 'Event-driven (trade invalidates position cache)'
    },
    
    'Database_Optimization': {
        'partitioning': 'By date (positions, metrics)',
        'indexing': 'Composite indexes on common queries',
        'materialized_views': 'Pre-aggregate desk/firm metrics',
        'read_replicas': '3 replicas for read-heavy loads'
    }
}
\`\`\`

**Bottom Line**: Risk platform architecture: 6 layers (data ingestion → position mgmt → risk calc → limit monitor → reporting → APIs). Technology: Kafka for messaging, Redis for cache, PostgreSQL for persistence, C++ for risk engine, React for UI. Data flow: trades → Kafka → position update → incremental VaR → limit check → alert → dashboard (<500ms). EOD: full VaR, stress tests, reports (30-60 min). Key decisions: real-time approximate vs batch accurate, horizontal scaling + caching, event-driven architecture. Handles 100K msg/sec, 10M positions, sub-second latency.`,
    },
    {
      question:
        'Implement a VaR calculation engine that supports multiple methodologies (historical, parametric, Monte Carlo). How do you optimize for both accuracy and performance, and what approximations are acceptable for real-time calculations?',
      answer: `VaR engine must balance accuracy with speed for different use cases:

**Multi-Method VaR Engine**
\`\`\`python
import numpy as np
from scipy import stats
import numba
from typing import List, Dict

class VaREngine:
    """
    Production VaR engine supporting multiple methodologies
    """
    
    def __init__(self, method='hybrid'):
        self.method = method
        self.cache = {}
        
    def calculate_var(self, portfolio, confidence=0.99, horizon=1):
        """
        Main entry point - selects method based on context
        """
        if self.method == 'historical':
            return self.historical_var(portfolio, confidence, horizon)
        elif self.method == 'parametric':
            return self.parametric_var(portfolio, confidence, horizon)
        elif self.method == 'monte_carlo':
            return self.monte_carlo_var(portfolio, confidence, horizon)
        elif self.method == 'hybrid':
            return self.hybrid_var(portfolio, confidence, horizon)
    
    def historical_var(self, portfolio, confidence, horizon):
        """
        Historical simulation - fast, no distribution assumptions
        
        Pros: Simple, no assumptions, handles fat tails
        Cons: Limited by historical data, backward-looking
        Speed: Fast (1-5 seconds for 1000 positions)
        """
        # Get historical returns
        returns = self.get_historical_returns(portfolio, lookback=252*3)  # 3 years
        
        # Calculate portfolio returns
        weights = portfolio.get_weights()
        portfolio_returns = returns @ weights
        
        # VaR = percentile
        var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        # Scale to horizon
        var_scaled = var * np.sqrt(horizon)
        
        return var_scaled
    
    def parametric_var(self, portfolio, confidence, horizon):
        """
        Parametric VaR (variance-covariance)
        
        Pros: Very fast, smooth estimates
        Cons: Assumes normality (underestimates tail risk)
        Speed: <1 second even for large portfolios
        Use: Real-time limit checks
        """
        # Get covariance matrix (cached, updated daily)
        cov_matrix = self.get_covariance_matrix(portfolio.assets)
        
        # Portfolio weights
        weights = portfolio.get_weights()
        
        # Portfolio variance
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # VaR assuming normal distribution
        z_score = stats.norm.ppf(confidence)
        var = z_score * portfolio_vol * np.sqrt(horizon)
        
        return var
    
    @numba.jit(nopython=True, parallel=True)
    def monte_carlo_var(self, portfolio, confidence, horizon, n_simulations=100000):
        """
        Monte Carlo VaR - most accurate
        
        Pros: Handles complex portfolios, non-linear risks, custom distributions
        Cons: Slow (minutes for complex portfolios)
        Speed: 10-60 seconds
        Use: EOD official VaR
        """
        # Simulate correlated returns
        returns = self.simulate_returns(
            portfolio,
            n_simulations=n_simulations,
            horizon=horizon
        )
        
        # Value portfolio under each scenario
        portfolio_values = []
        for scenario_returns in returns:
            value = self.value_portfolio(portfolio, scenario_returns)
            portfolio_values.append(value)
        
        # Calculate P&L
        current_value = portfolio.current_value()
        pnls = np.array(portfolio_values) - current_value
        
        # VaR = percentile of loss distribution
        var = -np.percentile(pnls, (1 - confidence) * 100)
        
        return var
    
    def hybrid_var(self, portfolio, confidence, horizon):
        """
        Hybrid approach: Trade off accuracy vs speed
        
        Strategy:
        - Linear instruments: Parametric (fast)
        - Options: Monte Carlo (accurate)
        - Aggregate carefully
        """
        linear_var = self.calculate_linear_portfolio_var(portfolio.linear_positions)
        option_var = self.calculate_option_var(portfolio.option_positions)
        
        # Aggregate (accounting for correlation)
        total_var = np.sqrt(
            linear_var**2 + 
            option_var**2 + 
            2 * self.correlation * linear_var * option_var
        )
        
        return total_var

# Performance comparison:
performance_comparison = {
    'Historical': {
        'time': '1-5 seconds',
        'accuracy': 'Good (historical tail)',
        'use': 'Intraday monitoring'
    },
    'Parametric': {
        'time': '<1 second',
        'accuracy': 'OK (underestimates tail)',
        'use': 'Real-time pre-trade checks'
    },
    'Monte_Carlo': {
        'time': '10-60 seconds',
        'accuracy': 'Best',
        'use': 'EOD official VaR'
    },
    'Hybrid': {
        'time': '2-10 seconds',
        'accuracy': 'Good compromise',
        'use': 'Intraday + complex portfolios'
    }
}
\`\`\`

**GPU Acceleration for Monte Carlo**
\`\`\`python
import torch

class GPUVaREngine:
    """
    GPU-accelerated Monte Carlo (50-100x faster)
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def monte_carlo_var_gpu(self, portfolio, n_simulations=1_000_000):
        """
        Run 1M simulations on GPU in seconds
        """
        # Generate correlated random numbers on GPU
        random_shocks = self.generate_correlated_shocks_gpu(
            n_assets=len(portfolio.assets),
            n_simulations=n_simulations,
            correlation_matrix=portfolio.correlation_matrix
        )
        
        # Portfolio valuation (parallel on GPU)
        portfolio_values = self.value_portfolio_gpu(
            portfolio.positions,
            portfolio.prices,
            random_shocks
        )
        
        # VaR calculation (on GPU)
        pnls = portfolio_values - portfolio.current_value
        var = -torch.quantile(pnls, 0.01).item()  # 99% VaR
        
        return var
    
    def generate_correlated_shocks_gpu(self, n_assets, n_simulations, correlation_matrix):
        """
        Cholesky decomposition on GPU
        """
        # Cholesky decomposition
        L = torch.linalg.cholesky(
            torch.tensor(correlation_matrix, device=self.device)
        )
        
        # Independent standard normal
        Z = torch.randn(n_assets, n_simulations, device=self.device)
        
        # Correlated shocks
        correlated_shocks = L @ Z
        
        return correlated_shocks

# Speed comparison:
# CPU Monte Carlo (100K sims): 30 seconds
# GPU Monte Carlo (1M sims): 3 seconds (10x more sims, 10x faster!)
\`\`\`

**Incremental VaR for Real-Time**
\`\`\`python
class IncrementalVaREngine:
    """
    Fast approximation for real-time updates
    
    Instead of recalculating full VaR after each trade,
    approximate change using marginal VaR
    """
    
    def __init__(self, portfolio):
        # Calculate full VaR once
        self.full_var = self.calculate_full_var(portfolio)
        self.portfolio = portfolio
        
        # Pre-calculate marginal VaRs
        self.marginal_vars = self.calculate_marginal_vars(portfolio)
    
    def update_var_for_trade(self, trade):
        """
        Fast update: ~10ms vs 10 seconds for full recalc
        """
        # Marginal VaR of this asset
        asset_id = trade.asset_id
        marginal_var = self.marginal_vars[asset_id]
        
        # Approximate VaR change
        # ΔVaR ≈ Marginal VaR × ΔPosition
        delta_var = marginal_var * trade.quantity
        
        # Update total VaR
        self.full_var += delta_var
        
        return self.full_var
    
    def calculate_marginal_vars(self, portfolio):
        """
        Marginal VaR = ∂VaR/∂position_i
        
        Pre-calculate for all assets
        """
        marginal_vars = {}
        
        for asset in portfolio.assets:
            # Sensitivity of VaR to $1 change in this asset
            # = (Cov matrix × weights) / portfolio_vol
            marginal_vars[asset.id] = self.calculate_marginal_var(asset, portfolio)
        
        return marginal_vars
    
    def recalibrate(self):
        """
        Periodically recalculate full VaR to prevent drift
        """
        self.full_var = self.calculate_full_var(self.portfolio)
        self.marginal_vars = self.calculate_marginal_vars(self.portfolio)

# Usage:
# - Trade happens → incremental update (10ms)
# - Every hour → full recalc (10s)
# - EOD → full Monte Carlo (60s)

# Accuracy: Incremental within 5% of full VaR
\`\`\`

**Acceptable Approximations**
\`\`\`python
approximations = {
    'Real_Time_(<1s)': {
        'method': 'Parametric or Incremental',
        'accuracy': '85-95% of true VaR',
        'acceptable': True,
        'reason': 'Need speed for pre-trade checks',
        'validation': 'Compare to full VaR hourly'
    },
    
    'Intraday_(<10s)': {
        'method': 'Historical or Hybrid',
        'accuracy': '90-98% of true VaR',
        'acceptable': True,
        'reason': 'Balance speed and accuracy',
        'validation': 'Compare to EOD VaR'
    },
    
    'EOD_Official': {
        'method': 'Monte Carlo (full)',
        'accuracy': '99%+',
        'acceptable': 'No approximations',
        'reason': 'Regulatory reporting, capital',
        'validation': 'Backtesting'
    }
}

# Key: Layer multiple methods
# Real-time: Fast approximate
# Hourly: Medium accuracy
# EOD: Full accurate
# Alert if methods diverge >10%
\`\`\`

**Bottom Line**: VaR engine supports 4 methods: (1) Historical (1-5s, good), (2) Parametric (<1s, fast but assumes normality), (3) Monte Carlo (10-60s, most accurate), (4) Hybrid (2-10s, balanced). GPU acceleration: 100x faster Monte Carlo (1M sims in 3s). Incremental VaR for real-time: marginal VaR × trade size (10ms vs 10s). Acceptable approximations: real-time 85-95% accurate (parametric/incremental), intraday 90-98% (historical/hybrid), EOD 99%+ (Monte Carlo). Strategy: layer methods—fast for real-time, accurate for EOD. Recalibrate hourly to prevent drift. Alert if methods diverge >10%.`,
    },
    {
      question:
        'Design the limit monitoring and breach handling system. How do you implement pre-trade compliance checks, real-time breach detection, escalation workflows, and audit trails for regulatory compliance?',
      answer: `Limit monitoring system must be fast, reliable, and auditable:

**Pre-Trade Compliance Architecture**
\`\`\`python
class PreTradeComplianceEngine:
    """
    Block trades that would breach limits BEFORE execution
    
    Critical: Must complete in <100ms (cant delay trading)
    """
    
    def __init__(self):
        # Load limits into Redis (in-memory for speed)
        self.limits = RedisLimitCache()
        
        # Load current usage
        self.usage = RedisUsageCache()
    
    def check_trade(self, trader_id, trade):
        """
        Pre-trade check: Will this trade breach limits?
        
        Returns: (approved: bool, reason: str, details: dict)
        """
        start_time = time.time()
        
        # Get all applicable limits
        limits = self.get_limits(trader_id)
        
        # Simulate trade impact
        hypothetical_usage = self.simulate_trade_impact(trader_id, trade)
        
        # Check each limit
        breaches = []
        for limit in limits:
            if hypothetical_usage[limit.type] > limit.value:
                breaches.append({
                    'limit_type': limit.type,
                    'limit_value': limit.value,
                    'current': self.usage.get(trader_id, limit.type),
                    'after_trade': hypothetical_usage[limit.type],
                    'breach_amount': hypothetical_usage[limit.type] - limit.value
                })
        
        # Record latency
        latency = time.time() - start_time
        self.record_latency(latency)  # Must be <100ms
        
        if breaches:
            return {
                'approved': False,
                'reason': 'Would breach limits',
                'breaches': breaches,
                'override_possible': self.check_override_eligibility(limits)
            }
        
        return {'approved': True}
    
    def simulate_trade_impact(self, trader_id, trade):
        """
        Fast approximation of post-trade risk metrics
        """
        # Current usage
        current = self.usage.get_all(trader_id)
        
        # Incremental impact (fast approximation)
        impact = {
            'var': current['var'] + self.estimate_var_impact(trade),
            'notional': current['notional'] + trade.notional,
            'concentration': self.estimate_concentration(trader_id, trade)
        }
        
        return impact

# Performance: 50-100ms per check
# Throughput: 1000 checks/second
\`\`\`

**Real-Time Breach Detection**
\`\`\`python
class RealTimeBreachDetector:
    """
    Continuous monitoring for limit breaches
    
    Checks every position update, market move
    """
    
    def __init__(self):
        # Subscribe to position updates
        self.kafka_consumer = KafkaConsumer('position_updates')
        
        # Alert queue
        self.alert_queue = KafkaProducer('alerts')
        
        # Breach history (deduplication)
        self.active_breaches = {}
    
    def monitor(self):
        """
        Continuous monitoring loop
        """
        for message in self.kafka_consumer:
            if message.type == 'POSITION_UPDATE':
                self.check_position_limits(message.trader_id)
            elif message.type == 'VAR_UPDATE':
                self.check_var_limits(message.trader_id)
            elif message.type == 'MARKET_MOVE':
                self.check_all_limits()  # Major market move → recheck all
    
    def check_var_limits(self, trader_id):
        """
        Check if VaR exceeds limit
        """
        # Get current VaR and limit
        current_var = self.get_current_var(trader_id)
        var_limit = self.get_limit(trader_id, 'VAR')
        
        if current_var > var_limit:
            # Calculate severity
            breach_pct = (current_var - var_limit) / var_limit
            
            if breach_pct > 0.10:  # >10% breach
                severity = 'CRITICAL'
            elif breach_pct > 0.05:  # 5-10% breach
                severity = 'HIGH'
            else:  # <5% breach
                severity = 'MEDIUM'
            
            # Check if already alerted (deduplication)
            if not self.is_active_breach(trader_id, 'VAR'):
                # New breach → send alert
                self.send_alert({
                    'trader_id': trader_id,
                    'limit_type': 'VAR',
                    'severity': severity,
                    'current': current_var,
                    'limit': var_limit,
                    'breach_pct': breach_pct,
                    'timestamp': datetime.now(),
                    'actions_required': self.get_required_actions(severity)
                })
                
                # Mark as active
                self.active_breaches[(trader_id, 'VAR')] = {
                    'first_breach': datetime.now(),
                    'severity': severity
                }

# Runs continuously, <1s latency from position update to alert
\`\`\`

**Escalation Workflow**
\`\`\`python
class BreachEscalationEngine:
    """
    Automatic escalation of unresolved breaches
    """
    
    def __init__(self):
        self.escalation_rules = {
            'MEDIUM': {
                '0_min': ['Trader'],
                '30_min': ['Trader', 'Desk_Head'],
                '60_min': ['Trader', 'Desk_Head', 'Risk_Manager']
            },
            'HIGH': {
                '0_min': ['Trader', 'Desk_Head'],
                '15_min': ['Trader', 'Desk_Head', 'Risk_Manager'],
                '30_min': ['Trader', 'Desk_Head', 'Risk_Manager', 'CRO']
            },
            'CRITICAL': {
                '0_min': ['Trader', 'Desk_Head', 'Risk_Manager', 'CRO'],
                '5_min': ['Trader', 'Desk_Head', 'Risk_Manager', 'CRO', 'CEO'],
                '10_min': ['All + Phone Call']
            }
        }
    
    def monitor_escalations(self):
        """
        Background job checking unresolved breaches
        """
        active_breaches = self.get_active_breaches()
        
        for breach in active_breaches:
            # Time since first breach
            duration = (datetime.now() - breach.timestamp).total_seconds() / 60
            
            # Get escalation level
            escalation = self.get_escalation_level(breach.severity, duration)
            
            # Check if we need to escalate further
            current_recipients = breach.alert_recipients
            required_recipients = escalation['recipients']
            
            new_recipients = set(required_recipients) - set(current_recipients)
            
            if new_recipients:
                # Escalate
                self.send_escalation_alert(breach, new_recipients)
                
                # Update breach record
                breach.alert_recipients = required_recipients
                breach.escalation_level += 1
                
                # Special actions
                if 'Phone Call' in escalation:
                    self.initiate_phone_call(breach.trader_id)
                
                # Log escalation
                self.audit_log.log_escalation(breach, escalation)

# Checks every 60 seconds
# Ensures breaches dont get ignored
\`\`\`

**Audit Trail for Compliance**
\`\`\`python
class AuditTrailSystem:
    """
    Comprehensive audit trail for regulators
    
    Must answer: Who did what, when, why?
    """
    
    def __init__(self):
        # Immutable append-only log
        self.audit_db = TimescaleDB('audit_trail')
    
    def log_limit_check(self, check):
        """
        Log every pre-trade compliance check
        """
        self.audit_db.insert({
            'event_type': 'PRETRADE_CHECK',
            'timestamp': datetime.now(),
            'trader_id': check.trader_id,
            'trade': check.trade,
            'approved': check.approved,
            'limits_checked': check.limits,
            'usage_before': check.usage_before,
            'usage_after': check.usage_after,
            'reason': check.reason,
            'latency_ms': check.latency
        })
    
    def log_breach(self, breach):
        """
        Log every limit breach
        """
        self.audit_db.insert({
            'event_type': 'BREACH',
            'timestamp': breach.timestamp,
            'trader_id': breach.trader_id,
            'limit_type': breach.limit_type,
            'severity': breach.severity,
            'limit_value': breach.limit_value,
            'actual_value': breach.actual_value,
            'breach_pct': breach.breach_pct,
            'alert_sent_to': breach.recipients,
            'market_conditions': self.capture_market_conditions()
        })
    
    def log_resolution(self, breach, resolution):
        """
        Log how breach was resolved
        """
        self.audit_db.insert({
            'event_type': 'BREACH_RESOLUTION',
            'timestamp': datetime.now(),
            'breach_id': breach.id,
            'resolution_type': resolution.type,  # 'REDUCED', 'OVERRIDE', 'MARKET_MOVE'
            'resolved_by': resolution.user_id,
            'actions_taken': resolution.actions,
            'time_to_resolve': resolution.duration,
            'final_usage': resolution.final_usage,
            'comments': resolution.comments
        })
    
    def generate_regulatory_report(self, start_date, end_date):
        """
        Generate report for regulators
        """
        return {
            'total_checks': self.count_checks(start_date, end_date),
            'total_breaches': self.count_breaches(start_date, end_date),
            'breach_rate': self.calculate_breach_rate(start_date, end_date),
            'avg_time_to_resolve': self.avg_resolution_time(start_date, end_date),
            'breaches_by_type': self.group_by_type(start_date, end_date),
            'breaches_by_trader': self.group_by_trader(start_date, end_date),
            'override_rate': self.calculate_override_rate(start_date, end_date),
            'repeated_breaches': self.find_repeated_breaches(start_date, end_date)
        }

# Retention: 7 years (regulatory requirement)
# Immutable: Cannot delete or modify (blockchain-like)
# Queryable: Fast queries for compliance reviews
\`\`\`

**Dashboard Integration**
\`\`\`python
# Real-time breach dashboard

breach_dashboard = {
    'Active_Breaches': {
        'display': 'List with severity color',
        'columns': [
            'Trader',
            'Limit Type',
            'Severity',
            'Breach %',
            'Duration',
            'Status'
        ],
        'actions': ['Acknowledge', 'View Details', 'Force Reduce'],
        'update': 'Real-time (WebSocket)'
    },
    
    'Breach_Timeline': {
        'display': 'Chart showing breaches over time',
        'breakdown': 'By severity, type, desk',
        'trend': 'Are breaches increasing?'
    },
    
    'Resolution_Metrics': {
        'avg_time_to_resolve': 'By severity',
        'resolution_types': 'Reduced vs Override vs Market',
        'repeated_offenders': 'Traders with >5 breaches/month'
    }
}
\`\`\`

**Bottom Line**: Limit monitoring: (1) Pre-trade compliance (<100ms, blocks trades that would breach), (2) Real-time detection (<1s from position update to alert), (3) Escalation (automatic escalation if unresolved—15/30/60 min by severity), (4) Audit trail (every check, breach, resolution logged immutably for regulators). Tech: Redis for limits/usage (speed), Kafka for alerts, TimescaleDB for audit (append-only, 7-year retention). Workflows: trader alert → escalate to desk head → risk manager → CRO → CEO. Dashboard: active breaches, timeline, resolution metrics. Regulatory ready: complete audit trail, cant be modified, fast queries for compliance reviews.`,
    },
  ],
} as const;
