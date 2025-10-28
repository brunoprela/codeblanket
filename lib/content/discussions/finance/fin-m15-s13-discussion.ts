export default {
  id: 'fin-m15-s13-discussion',
  title: 'Real-Time Risk Monitoring - Discussion Questions',
  questions: [
    {
      question:
        'Explain the architecture requirements for real-time risk monitoring systems. How do firms achieve sub-second latency for VaR calculations across thousands of positions, and what are the tradeoffs between accuracy and speed?',
      answer: `Real-time risk monitoring requires specialized architecture for speed and scale:

**Latency Requirements**
\`\`\`python
latency_requirements = {
    'Pre-trade compliance': '<100ms (before order sent)',
    'Position updates': '<500ms (after trade)',
    'VaR calculation': '<1 second (for limit checks)',
    'Risk reports': '<5 seconds (trader dashboard)',
    'Stress testing': '<30 seconds (on-demand)'
}

# Challenge: Portfolio with 10,000 positions
# Full VaR calculation could take minutes
# Need approximations for real-time
\`\`\`

**System Architecture**
\`\`\`python
class RealTimeRiskSystem:
    """
    Real-time risk monitoring architecture
    """
    
    def __init__(self):
        self.components = {
            '1_data_feed': 'Market data (real-time prices)',
            '2_position_cache': 'In-memory position store',
            '3_risk_engine': 'Calculation engine',
            '4_limit_monitor': 'Limit checking',
            '5_alert_system': 'Real-time alerts',
            '6_dashboard': 'Trader/risk manager UI'
        }
    
    # Architecture pattern
    architecture = {
        'Market Data': {
            'source': 'Bloomberg, Reuters, exchanges',
            'protocol': 'FIX, WebSocket',
            'latency': '10-100ms',
            'rate': '1000s updates/second'
        },
        
        'Position Cache': {
            'technology': 'Redis, Hazelcast (in-memory)',
            'update': 'Real-time from OMS',
            'access': '<1ms lookup',
            'capacity': 'Millions of positions'
        },
        
        'Risk Engine': {
            'technology': 'C++, FPGA for speed',
            'parallelization': 'GPU, multi-core',
            'approximation': 'Incremental VaR',
            'throughput': '10K VaR calcs/second'
        },
        
        'Alert System': {
            'technology': 'Kafka (message queue)',
            'destinations': 'Bloomberg MSG, email, SMS',
            'latency': '<1 second end-to-end'
        }
    }

# Stack example:
tech_stack = {
    'data_ingestion': 'Kafka, Solace',
    'computation': 'C++, Rust (speed)',
    'storage': 'Redis (cache), PostgreSQL (persistent)',
    'messaging': 'Kafka, RabbitMQ',
    'ui': 'React, WebSocket (real-time updates)'
}
\`\`\`

**Fast VaR Calculation: Incremental Updates**
\`\`\`python
class IncrementalVaRCalculator:
    """
    Instead of recalculating full portfolio VaR,
    approximate change from trade
    """
    
    def __init__(self, portfolio):
        # Calculate full VaR once
        self.full_var = self.calculate_full_var(portfolio)
        self.portfolio = portfolio
    
    def update_for_trade(self, new_trade):
        """
        Fast approximation: Marginal VaR
        
        ΔVaR ≈ Marginal VaR × Trade Size
        
        Much faster than full recalculation
        """
        # Marginal VaR of this asset
        marginal_var = self.get_marginal_var(new_trade.asset)
        
        # Approximate VaR change
        delta_var = marginal_var * new_trade.size
        
        # Update total VaR
        self.full_var += delta_var
        
        return self.full_var
    
    # Recalculate full VaR periodically (every hour, EOD)
    # Prevents drift from approximations

# Example:
# Full VaR calculation: 10 seconds
# Incremental update: 10 milliseconds (1000x faster!)
\`\`\`

**Accuracy vs Speed Tradeoffs**
\`\`\`python
calculation_methods = {
    'Full Monte Carlo VaR': {
        'accuracy': 'High (100%)',
        'time': '10-60 seconds',
        'frequency': 'EOD, overnight',
        'use': 'Official risk reporting'
    },
    
    'Historical VaR': {
        'accuracy': 'Medium (95%)',
        'time': '1-5 seconds',
        'frequency': 'Intraday (hourly)',
        'use': 'Intraday monitoring'
    },
    
    'Parametric VaR': {
        'accuracy': 'Medium (90%)',
        'time': '0.1-1 seconds',
        'frequency': 'Real-time',
        'use': 'Pre-trade checks, dashboards'
    },
    
    'Incremental VaR': {
        'accuracy': 'Low-Medium (85%)',
        'time': '<0.1 seconds',
        'frequency': 'Every trade',
        'use': 'Real-time limit monitoring'
    }
}

# Strategy: Layer multiple approaches
# - Real-time: Incremental (fast, approximate)
# - Hourly: Historical (balance)
# - EOD: Monte Carlo (slow, accurate)
# - Alert if methods diverge >10%
\`\`\`

**Scaling for Volume**
\`\`\`python
# Large firm: 1M positions, 100K trades/day

scaling_strategies = {
    '1. Hierarchical Aggregation': {
        'approach': 'Pre-aggregate at desk/book level',
        'benefit': 'Only recalculate affected sub-portfolios',
        'example': 'Trade in equity desk → only recalc equity VaR',
        'speedup': '100x'
    },
    
    '2. GPU Acceleration': {
        'approach': 'Use GPUs for Monte Carlo',
        'benefit': 'Parallel simulation (1000s at once)',
        'speedup': '50-100x vs CPU'
    },
    
    '3. Caching': {
        'approach': 'Cache sensitivities (Greeks)',
        'benefit': 'Dont recalculate unchanged positions',
        'speedup': '10x'
    },
    
    '4. Sampling': {
        'approach': 'For real-time, use representative sample',
        'benefit': 'VaR on 1000 positions instead of 100K',
        'accuracy': '95% (good enough for real-time)'
    }
}
\`\`\`

**Example Implementation**
\`\`\`python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FastRiskEngine:
    def __init__(self):
        self.cache = {}
        self.gpu_available = torch.cuda.is_available()
    
    def calculate_var_realtime(self, portfolio):
        """
        Real-time VaR using approximations
        """
        # Check cache (5-second TTL)
        if self.cache_valid(portfolio.id):
            return self.cache[portfolio.id]
        
        # Hierarchical: Calculate per desk in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            desk_vars = list(executor.map(
                self.calculate_desk_var,
                portfolio.desks
            ))
        
        # Aggregate (accounting for correlation)
        total_var = self.aggregate_vars(desk_vars, portfolio.correlation_matrix)
        
        # Cache result
        self.cache[portfolio.id] = (total_var, time.time())
        
        return total_var
    
    def calculate_desk_var(self, desk):
        """
        Desk-level VaR (parametric for speed)
        """
        # Get sensitivities (cached)
        sensitivities = self.get_sensitivities(desk)
        
        # Parametric VaR
        portfolio_vol = np.sqrt(
            sensitivities.T @ desk.covariance_matrix @ sensitivities
        )
        
        var = 2.33 * portfolio_vol  # 99% VaR
        
        return var

# Latency: <100ms for 100K position portfolio
\`\`\`

**Bottom Line**: Real-time risk monitoring requires specialized architecture: in-memory caching (Redis), fast computation (C++/GPU), incremental calculations, hierarchical aggregation. Tradeoff: accuracy vs speed. Solution: layer multiple methods—incremental for real-time (<100ms), parametric for intraday (<1s), Monte Carlo for EOD (accurate). Scaling techniques: hierarchical aggregation, GPU parallelization, caching, sampling. Achieves sub-second VaR across 100K+ positions by trading perfect accuracy for speed at the real-time layer.`,
    },
    {
      question:
        'Describe WebSocket-based real-time risk dashboards. What metrics should be displayed to traders vs risk managers, and how do you design alerts that inform without overwhelming?',
      answer: `Real-time dashboards must balance information richness with cognitive load:

**WebSocket Architecture**
\`\`\`python
# WebSocket provides bidirectional real-time communication

class RiskDashboardWebSocket:
    def __init__(self, user_role):
        self.ws = websocket.create_connection('wss://risk.firm.com')
        self.role = user_role
        self.subscribe_to_relevant_feeds()
    
    def subscribe_to_relevant_feeds(self):
        if self.role == 'TRADER':
            feeds = [
                'my_positions',
                'my_pnl',
                'my_limits',
                'market_data'
            ]
        elif self.role == 'RISK_MANAGER':
            feeds = [
                'desk_aggregates',
                'firm_var',
                'limit_breaches',
                'stress_tests'
            ]
        
        for feed in feeds:
            self.ws.send(json.dumps({'subscribe': feed}))
    
    def on_message(self, message):
        """
        Real-time updates pushed from server
        """
        data = json.parse(message)
        
        if data['type'] == 'var_update':
            self.update_var_gauge(data['var'], data['limit'])
        elif data['type'] == 'pnl_update':
            self.update_pnl_chart(data['pnl'])
        elif data['type'] == 'alert':
            self.show_alert(data)

# Update frequency: 
# - P&L: Every trade (seconds)
# - VaR: Every 5 seconds
# - Positions: Every trade
# - Limits: Every 5 seconds
\`\`\`

**Trader Dashboard Design**
\`\`\`python
trader_dashboard = {
    'Top Priority (Always Visible)': {
        'pnl_today': {
            'display': 'Large number, color-coded',
            'green': 'Positive',
            'red': 'Negative',
            'update': 'Real-time (every trade)'
        },
        
        'var_vs_limit': {
            'display': 'Gauge (speedometer style)',
            'green': '0-70% of limit',
            'yellow': '70-90% of limit',
            'red': '>90% of limit',
            'update': 'Every 5 seconds'
        },
        
        'top_positions': {
            'display': 'Top 10 by P&L contribution',
            'columns': ['Symbol', 'Size', 'P&L', '% of VaR'],
            'update': 'Every 30 seconds'
        }
    },
    
    'Secondary (Tabs/Expandable)': {
        'greeks_summary': {
            'delta': 'Net delta exposure',
            'gamma': 'Curvature risk',
            'vega': 'Vol exposure',
            'theta': 'Time decay'
        },
        
        'stress_scenarios': {
            'rates_up_100bp': 'Impact if rates +1%',
            'equity_down_10': 'Impact if stocks -10%',
            'vol_spike': 'Impact if VIX +10'
        },
        
        'historical_pnl': {
            'chart': 'Intraday P&L chart',
            'mtd': 'Month-to-date',
            'ytd': 'Year-to-date'
        }
    }
}

# Key principle: Information hierarchy
# Most important = largest, top of screen
# Less urgent = tabs, drill-down
\`\`\`

**Risk Manager Dashboard Design**
\`\`\`python
risk_manager_dashboard = {
    'Overview (Heat Map)': {
        'display': 'Desk heat map',
        'axes': 'Desk × Risk Metric',
        'color': {
            'green': 'Low utilization, positive P&L',
            'yellow': 'High utilization',
            'red': 'Breach or large loss'
        },
        'update': 'Every 10 seconds'
    },
    
    'Firm Aggregates': {
        'total_var': 'Firm-wide VaR vs limit',
        'total_pnl': 'Firm-wide P&L',
        'concentration': 'Top 10 exposures',
        'liquidity': 'LCR, available margin'
    },
    
    'Exceptions (Alerts)': {
        'limit_breaches': 'Active breaches requiring action',
        'large_losses': 'Desks down >$5M',
        'correlation_breaks': 'Unusual correlations',
        'system_issues': 'Risk calculation failures'
    },
    
    'Drill-Down': {
        'click_desk': 'Zoom to desk-level detail',
        'click_trader': 'See trader positions',
        'click_alert': 'Full context + history'
    }
}

# Key principle: Exception-based
# Only show what needs attention
# Heat map for scanning, drill-down for detail
\`\`\`

**Alert Design: Preventing Overload**
\`\`\`python
# Problem: Too many alerts → ignored ("alert fatigue")

alert_prioritization = {
    'Critical (Red)': {
        'criteria': [
            'Hard limit breach',
            'Loss >$10M',
            'System failure',
            'Regulatory breach'
        ],
        'delivery': [
            'Dashboard (pop-up)',
            'SMS',
            'Bloomberg MSG',
            'Phone call (if not acknowledged)'
        ],
        'frequency': 'Immediate, escalating'
    },
    
    'Important (Yellow)': {
        'criteria': [
            'Soft limit breach',
            'Loss >$5M',
            'High VaR utilization (>80%)',
            'Concentration risk'
        ],
        'delivery': [
            'Dashboard',
            'Email',
            'Bloomberg MSG'
        ],
        'frequency': 'Once, reminder after 30 min'
    },
    
    'Informational (Blue)': {
        'criteria': [
            'Loss >$1M',
            'VaR utilization >70%',
            'Position size increase'
        ],
        'delivery': [
            'Dashboard only'
        ],
        'frequency': 'Batched (hourly summary)'
    }
}

# Alert fatigue prevention:
alert_management = {
    'Deduplication': 'Same alert only once per hour',
    'Aggregation': 'Batch low-priority alerts',
    'Escalation': 'Auto-escalate if not acknowledged',
    'Snooze': 'Allow temporary suppression',
    'Tuning': 'Adjust thresholds based on false positive rate'
}
\`\`\`

**Example Alert Implementation**
\`\`\`python
class SmartAlertSystem:
    def __init__(self):
        self.alert_history = {}
        self.acknowledged = {}
    
    def should_send_alert(self, alert):
        """
        Decide if alert should be sent
        """
        # Check if duplicate
        if self.is_duplicate(alert):
            return False
        
        # Check if recently sent
        if self.recently_sent(alert, window=3600):  # 1 hour
            return False
        
        # Check if acknowledged
        if alert.id in self.acknowledged:
            return False
        
        return True
    
    def send_alert(self, alert):
        """
        Send via appropriate channels
        """
        if alert.priority == 'CRITICAL':
            self.send_sms(alert)
            self.send_bloomberg(alert)
            self.show_popup(alert)
            
            # Escalate if not acknowledged in 5 minutes
            self.schedule_escalation(alert, delay=300)
        
        elif alert.priority == 'IMPORTANT':
            self.send_email(alert)
            self.send_bloomberg(alert)
            self.show_notification(alert)
        
        else:  # INFORMATIONAL
            self.add_to_dashboard(alert)
        
        # Record
        self.alert_history[alert.id] = time.time()
    
    def escalate(self, alert):
        """
        Escalate unacknowledged critical alerts
        """
        if alert.id not in self.acknowledged:
            # Send to manager
            self.send_alert_to_manager(alert)
            
            # Phone call after 10 minutes
            if time.time() - alert.time > 600:
                self.initiate_phone_call(alert.owner)

# Example alert
alert = {
    'type': 'LIMIT_BREACH',
    'severity': 'CRITICAL',
    'trader': 'John Doe',
    'limit_type': 'VaR',
    'current': 12_000_000,
    'limit': 10_000_000,
    'breach_pct': 0.20,
    'message': 'VaR limit breached by 20%',
    'recommended_action': 'Reduce position immediately',
    'context': {
        'recent_trades': [...],
        'position_breakdown': {...}
    }
}
\`\`\`

**Visualization Best Practices**
\`\`\`python
visualization_principles = {
    'Color': {
        'red': 'Danger/loss/breach (sparingly)',
        'green': 'Good/profit/safe',
        'yellow': 'Warning',
        'gray': 'Neutral',
        'avoid': 'Too many colors (confusing)'
    },
    
    'Size': {
        'large': 'Most important metrics',
        'small': 'Details',
        'principle': 'Size = importance'
    },
    
    'Updates': {
        'smooth_animation': 'Numbers transition smoothly',
        'highlight_changes': 'Flash briefly when updated',
        'avoid_seizure': 'Dont flash constantly'
    },
    
    'Density': {
        'trader': 'Dense (they want detail)',
        'executive': 'Sparse (high-level only)',
        'principle': 'Match density to user need'
    }
}

# Example: VaR gauge
var_gauge = {
    'type': 'Semi-circular speedometer',
    'ranges': [
        '0-70%: Green',
        '70-90%: Yellow',
        '90-100%: Orange',
        '>100%: Red + flashing'
    ],
    'needle': 'Shows current utilization',
    'number': 'Large display of %, dollar amount smaller'
}
\`\`\`

**Bottom Line**: Real-time dashboards use WebSocket for instant updates. Trader dashboard: focus on P&L, VaR utilization, top positions (information they can act on). Risk manager dashboard: heat map for scanning, exception-based (only show problems), drill-down for detail. Alert design critical: prioritize (red/yellow/blue), deduplicate, aggregate, escalate if ignored. Prevent alert fatigue—only send actionable alerts. Visualization: color = urgency, size = importance, smooth updates, appropriate density for audience.`,
    },
    {
      question:
        'Explain how to monitor and respond to intraday P&L moves. What P&L attribution should be performed in real-time, and how do firms distinguish between expected P&L (market moves) and unexplained P&L (potential errors)?',
      answer: `Real-time P&L monitoring separates signal from noise to catch errors fast:

**P&L Components**
\`\`\`python
total_pnl = explained_pnl + unexplained_pnl

explained_pnl = {
    'market_moves': 'Positions × Price changes',
    'new_trades': 'Trade P&L',
    'time_decay': 'Theta (options)',
    'carry': 'Coupon, dividends, funding'
}

unexplained_pnl = {
    'model_changes': 'Valuation model updates',
    'missing_trades': 'Trade not booked',
    'pricing_errors': 'Wrong price',
    'system_issues': 'Calculation bugs'
}

# Goal: Unexplained should be ~0%
# If >2% → investigate immediately
\`\`\`

**Real-Time P&L Attribution**
\`\`\`python
class RealTimePnLAttribution:
    def calculate_explained_pnl(self, positions, price_changes):
        """
        Attribute P&L to risk factors
        """
        explained = {}
        
        # Delta P&L (linear)
        explained['delta'] = positions.delta @ price_changes
        
        # Gamma P&L (convexity)
        explained['gamma'] = 0.5 * positions.gamma @ (price_changes ** 2)
        
        # Vega P&L (volatility)
        explained['vega'] = positions.vega @ vol_changes
        
        # Theta P&L (time decay)
        explained['theta'] = positions.theta * time_elapsed
        
        # Carry P&L (coupons, dividends)
        explained['carry'] = self.calculate_carry(positions)
        
        # New trades P&L
        explained['new_trades'] = self.calculate_trade_pnl()
        
        return explained
    
    def calculate_unexplained(self, actual_pnl, explained_pnl):
        """
        Residual P&L requiring investigation
        """
        unexplained = actual_pnl - sum(explained_pnl.values())
        
        if abs(unexplained) / abs(actual_pnl) > 0.02:  # >2%
            self.alert_unexplained_pnl(unexplained)
        
        return unexplained

# Example output (intraday):
pnl_attribution = {
    'total_pnl': 5_200_000,
    
    'explained': {
        'delta': 4_500_000,      # Stocks up
        'gamma': 300_000,        # Convexity gain
        'vega': 200_000,         # Vol up
        'theta': -100_000,       # Time decay
        'carry': 50_000,         # Dividends
        'new_trades': 150_000   # Trade P&L
    },
    'explained_total': 5_100_000,
    
    'unexplained': 100_000,  # $100K (1.9% → acceptable)
    
    'status': 'GREEN'
}
\`\`\`

**Unexplained P&L Thresholds**
\`\`\`python
unexplained_thresholds = {
    'Green (<1%)': {
        'meaning': 'Normal (rounding, timing)',
        'action': 'Monitor',
        'example': '$50K unexplained on $5M total'
    },
    
    'Yellow (1-5%)': {
        'meaning': 'Elevated but manageable',
        'action': 'Investigate but not urgent',
        'example': '$200K unexplained on $5M total',
        'causes': [
            'Model approximations',
            'Cross effects',
            'Timing differences'
        ]
    },
    
    'Red (>5%)': {
        'meaning': 'Significant issue',
        'action': 'Investigate immediately',
        'example': '$500K unexplained on $5M total',
        'causes': [
            'Missing trades',
            'Wrong prices',
            'System errors',
            'Fraud'
        ]
    }
}
\`\`\`

**Real-World Error Examples**
\`\`\`python
# Example 1: Missing trade
error_1 = {
    'symptom': 'Large unexplained P&L',
    'investigation': {
        'actual_pnl': -2_000_000,  # Down $2M
        'explained_pnl': +500_000,  # Model says up $500K
        'unexplained': -2_500_000   # $2.5M gap!
    },
    'cause': 'Sold $50M stock, trade not booked in system',
    'resolution': 'Book trade, unexplained drops to ~0%',
    'lesson': 'Check trade confirmations in real-time'
}

# Example 2: Wrong price
error_2 = {
    'symptom': 'Unexplained P&L on single position',
    'investigation': {
        'position_pnl': +1_000_000,  # Position up $1M
        'expected_pnl': +100_000,    # Should be +$100K
        'unexplained': +900_000      # $900K too high
    },
    'cause': 'Pricing error: Bond priced at 105 instead of 95',
    'resolution': 'Correct price, unexplained resolved',
    'lesson': 'Validate prices against external sources'
}

# Example 3: Model change
error_3 = {
    'symptom': 'Large unexplained P&L overnight',
    'investigation': {
        'yesterday_eod': 10_000_000,
        'today_open': 11_000_000,  # +$1M overnight?
        'explained': 0  # No market moves
    },
    'cause': 'Valuation model updated (new methodology)',
    'resolution': 'Classify as model P&L, not unexplained',
    'lesson': 'Track model changes separately'
}
\`\`\`

**Monitoring Dashboard**
\`\`\`python
intraday_pnl_dashboard = {
    'Top Section': {
        'total_pnl': {
            'display': 'Large number',
            'breakdown': 'Click to see attribution',
            'trend': 'Intraday chart'
        },
        
        'unexplained_pnl': {
            'display': 'Number + % of total',
            'color': 'Green/Yellow/Red by threshold',
            'alert': 'Automatic if red'
        }
    },
    
    'Attribution Table': {
        'columns': ['Factor', 'P&L', '% of Total'],
        'rows': [
            'Delta',
            'Gamma',
            'Vega',
            'Theta',
            'Carry',
            'New trades',
            'Unexplained'
        ],
        'sortable': True,
        'drilldown': 'Click to see positions'
    },
    
    'Largest Movers': {
        'display': 'Top 10 positions by P&L',
        'purpose': 'Quickly see drivers',
        'columns': ['Position', 'P&L', 'Explained', 'Unexplained']
    },
    
    'Alerts': {
        'high_unexplained': 'Position with >10% unexplained',
        'pricing_stale': 'Price not updated in 10 min',
        'missing_trade': 'Trade confirmed but not in system'
    }
}
\`\`\`

**Automated Checks**
\`\`\`python
class PnLValidation:
    def run_checks_realtime(self):
        """
        Automated sanity checks on P&L
        """
        checks = {
            'Total P&L vs Sum of Positions': {
                'test': 'Do position P&Ls add up to total?',
                'tolerance': '$10K',
                'alert_if_fails': True
            },
            
            'P&L vs Market Moves': {
                'test': 'If SPX +1%, is P&L consistent with beta?',
                'tolerance': '20%',
                'alert_if_fails': True
            },
            
            'Price Staleness': {
                'test': 'All prices updated in last 5 min?',
                'alert_if_stale': True
            },
            
            'Trade Confirmations': {
                'test': 'All executed trades booked?',
                'check_interval': '1 minute',
                'alert_if_missing': True
            },
            
            'Limit Consistency': {
                'test': 'P&L within expected range given VaR?',
                'threshold': '3-sigma move',
                'alert_if_breach': True
            }
        }
        
        for check in checks.values():
            result = self.run_check(check)
            if not result.passed:
                self.alert(check, result)

# Run every 60 seconds
\`\`\`

**Response Procedures**
\`\`\`python
unexplained_pnl_procedures = {
    'Step 1: Immediate Check (1 minute)': [
        'Verify prices (compare to external source)',
        'Check for missing trades (query OMS)',
        'Verify position quantities (inventory check)'
    ],
    
    'Step 2: Detailed Investigation (10 minutes)': [
        'Position-level attribution',
        'Compare yesterday EOD to today start',
        'Check for model changes',
        'Review recent trades',
        'Check system logs for errors'
    ],
    
    'Step 3: Escalation (if unresolved in 30 min)': [
        'Notify risk manager',
        'Engage technology (if system issue)',
        'Consider stopping trading (if material)'
    ],
    
    'Step 4: Documentation': [
        'Log investigation',
        'Record resolution',
        'Update procedures if new issue type'
    ]
}
\`\`\`

**Bottom Line**: Real-time P&L attribution separates explained (delta, gamma, vega, theta, carry, trades) from unexplained (errors). Unexplained should be <2%. Thresholds: <1% green, 1-5% yellow, >5% red (investigate immediately). Common causes of unexplained: missing trades, wrong prices, model changes, system errors. Automated checks: total vs sum, staleness, trade confirmations, limit consistency. Fast response critical—errors can compound. Dashboard shows total, attribution, unexplained %, largest movers. Alert on red unexplained, investigate within minutes.`,
    },
  ],
} as const;
