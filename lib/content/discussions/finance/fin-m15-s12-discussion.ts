export default {
  id: 'fin-m15-s12-discussion',
  title: 'Position Limits and Risk Limits - Discussion Questions',
  questions: [
    {
      question:
        'Explain the hierarchy of risk limits in a trading organization (firm-wide, desk-level, trader-level). How are limits set, monitored, and enforced, and what happens when a limit is breached?',
      answer: `Risk limits create a cascading control framework from firm to individual:

**Limit Hierarchy**
\`\`\`python
risk_limit_hierarchy = {
    'Board of Directors': {
        'sets': 'Risk appetite statement',
        'example': 'Maximum firm-wide VaR = $200M',
        'frequency': 'Annual review'
    },
    
    'CRO / Risk Committee': {
        'sets': 'Firm-wide limits by risk type',
        'limits': {
            'total_var': 200_000_000,
            'stress_var': 500_000_000,
            'credit_exposure': 50_000_000_000,
            'liquidity_buffer': 10_000_000_000
        },
        'frequency': 'Quarterly review'
    },
    
    'Business Unit Heads': {
        'allocate': 'Limits to desks',
        'equities': {
            'var': 80_000_000,  # 40% of firm
            'stress_var': 200_000_000
        },
        'fixed_income': {
            'var': 70_000_000,  # 35% of firm
            'stress_var': 180_000_000
        },
        'frequency': 'Monthly review'
    },
    
    'Desk Heads': {
        'allocate': 'Limits to traders',
        'trader_1': {
            'var': 10_000_000,
            'position_limit': 100_000_000,  # Notional
            'loss_limit': 2_000_000  # Daily loss
        },
        'trader_2': {
            'var': 8_000_000,
            'position_limit': 80_000_000,
            'loss_limit': 1_500_000
        },
        'frequency': 'Weekly review'
    }
}

# Sum of desk limits ≤ firm limit (with buffer)
# Sum of trader limits ≤ desk limit
\`\`\`

**How Limits Are Set**

\`\`\`python
def allocate_limits(firm_limit, desk_performance, desk_capacity):
    """
    Allocate firm limit to desks based on:
    1. Historical performance (return/risk)
    2. Capacity (can they use it?)
    3. Strategy risk profile
    4. Regulatory constraints
    """
    
    # Example: Firm VaR limit = $200M
    
    desk_allocations = {
        'Equities': {
            'sharpe_ratio': 1.5,  # Best performance
            'var_utilization': 0.90,  # Uses limits well
            'allocation': 80_000_000,  # Gets most
            'pct': 0.40
        },
        
        'Fixed Income': {
            'sharpe_ratio': 1.2,
            'var_utilization': 0.85,
            'allocation': 70_000_000,
            'pct': 0.35
        },
        
        'Commodities': {
            'sharpe_ratio': 0.8,  # Weaker performance
            'var_utilization': 0.60,  # Doesnt use limit
            'allocation': 30_000_000,  # Gets less
            'pct': 0.15
        },
        
        'Buffer': {
            'reason': 'Unexpected opportunities, intraday spikes',
            'allocation': 20_000_000,
            'pct': 0.10
        }
    }
    
    return desk_allocations

# Reallocate quarterly based on performance
\`\`\`

**Monitoring System**
\`\`\`python
class RiskLimitMonitor:
    def __init__(self):
        self.limits = self.load_limits()
        self.positions = self.get_positions()
        
    def check_limits_realtime(self):
        """
        Check every trade against limits
        Real-time (pre-trade and post-trade)
        """
        for trader in self.traders:
            # Calculate current usage
            current_var = self.calculate_var(trader.positions)
            limit_var = self.limits[trader.id]['var']
            
            # Check utilization
            utilization = current_var / limit_var
            
            if utilization > 1.0:
                # BREACH!
                self.alert_breach(trader, 'VAR', current_var, limit_var)
            elif utilization > 0.90:
                # Warning (approaching limit)
                self.alert_warning(trader, 'VAR', current_var, limit_var)
            elif utilization > 0.80:
                # Monitor (elevated usage)
                self.monitor(trader, 'VAR', current_var, limit_var)
    
    def pre_trade_check(self, trader, new_trade):
        """
        Before trade executed, check if would breach limit
        """
        current_var = self.calculate_var(trader.positions)
        hypothetical_var = self.calculate_var(
            trader.positions + [new_trade]
        )
        
        if hypothetical_var > self.limits[trader.id]['var']:
            return {
                'approved': False,
                'reason': 'Would breach VaR limit',
                'current': current_var,
                'after_trade': hypothetical_var,
                'limit': self.limits[trader.id]['var']
            }
        
        return {'approved': True}

# Real-time monitoring (sub-second latency)
# Automated alerts via email, Bloomberg, SMS
\`\`\`

**Breach Response**
\`\`\`python
breach_procedures = {
    'Soft Breach (80-100%)': {
        'action': 'Warning to trader',
        'notification': 'Desk head',
        'requirement': 'Acknowledge warning',
        'follow_up': 'Monitor closely'
    },
    
    'Hard Breach (100-110%)': {
        'action': 'Stop new risk-increasing trades',
        'notification': 'Desk head + Risk Manager',
        'requirement': 'Explain breach within 1 hour',
        'follow_up': 'Plan to reduce within 24 hours'
    },
    
    'Material Breach (>110%)': {
        'action': 'Immediate position reduction required',
        'notification': 'CRO + Business Unit Head',
        'requirement': 'Reduce below limit within 4 hours',
        'follow_up': 'Written explanation + disciplinary review',
        'consequences': 'Bonus reduction, limit cut, termination'
    }
}

# Example breach
breach_example = {
    'trader': 'Trader A',
    'limit': 10_000_000,
    'current': 12_000_000,  # 120% (material breach)
    'breach_pct': 0.20,
    
    'response': [
        '10:00am: Breach detected',
        '10:01am: Auto-block new trades',
        '10:02am: Alert to trader, desk head, risk',
        '10:30am: Trader explains (market moved)',
        '11:00am: Forced reduction plan approved',
        '2:00pm: Reduced to $9.5M (below limit)',
        '5:00pm: Post-mortem report',
        'Next day: Limit reduced to $8M (penalty)'
    ]
}
\`\`\`

**Limit Types**

\`\`\`python
limit_types = {
    'VaR Limit': {
        'measures': 'Potential loss (99% confidence)',
        'typical': '$10M VaR for senior trader',
        'frequency': 'Daily'
    },
    
    'Stress VaR': {
        'measures': 'Loss in stressed scenario',
        'typical': '2-3x VaR limit',
        'frequency': 'Daily'
    },
    
    'Position Limit': {
        'measures': 'Notional exposure',
        'typical': '$100M notional',
        'frequency': 'Real-time'
    },
    
    'Loss Limit': {
        'measures': 'Actual P&L loss',
        'typical': '$2M daily, $10M monthly',
        'frequency': 'Intraday',
        'action': 'Auto-stop trading if hit'
    },
    
    'Concentration Limit': {
        'measures': 'Single name/sector exposure',
        'typical': 'No more than 20% in one stock',
        'frequency': 'Daily'
    },
    
    'DV01 Limit': {
        'measures': 'Interest rate sensitivity',
        'typical': '$100K DV01',
        'frequency': 'Real-time'
    }
}
\`\`\`

**Bottom Line**: Limits cascade from board to trader, ensuring firm-wide risk stays within appetite. Set based on performance and capacity. Monitored real-time with pre-trade checks. Breaches trigger automatic escalation: soft (warning), hard (stop new trades), material (force reduction). Comprehensive limit framework essential for controlling risk in decentralized trading organization.`,
    },
    {
      question:
        'Explain the difference between hard limits (cannot be breached) and soft limits (can be breached with approval). When should each be used, and how do override procedures work?',
      answer: `Hard vs soft limits balance control with business flexibility:

**Hard Limits (Cannot Be Breached)**
\`\`\`python
class HardLimit:
    """
    System enforced - no override possible
    Trade rejected automatically
    """
    
    def pre_trade_check(self, trader, trade):
        if self.would_breach(trader, trade):
            return {
                'status': 'REJECTED',
                'message': 'Hard limit breach - trade blocked',
                'override': 'Not possible'
            }
        return {'status': 'APPROVED'}

# Examples of hard limits:
hard_limits = {
    'Regulatory': {
        'example': 'Position limit set by regulator',
        'reason': 'Cannot violate law',
        'consequence': 'Regulatory breach → fines'
    },
    
    'Risk Capacity': {
        'example': 'Firm total VaR = $200M (all capital at risk)',
        'reason': 'Beyond risk capacity → insolvency risk',
        'consequence': 'Existential threat'
    },
    
    'Client Mandate': {
        'example': 'Pension fund: no leverage',
        'reason': 'Contractual obligation',
        'consequence': 'Breach of fiduciary duty'
    },
    
    'Credit Line': {
        'example': 'Prime broker line = $1B',
        'reason': 'Cannot trade beyond credit facility',
        'consequence': 'Trade will fail to settle'
    }
}
\`\`\`

**Soft Limits (Can Be Breached with Approval)**
\`\`\`python
class SoftLimit:
    """
    Requires human judgment
    Can be overridden with approval
    """
    
    def pre_trade_check(self, trader, trade):
        if self.would_breach(trader, trade):
            return {
                'status': 'REQUIRES_APPROVAL',
                'approvers': ['Desk Head', 'Risk Manager'],
                'reason': 'Soft limit breach',
                'override_possible': True
            }
        return {'status': 'APPROVED'}

# Examples of soft limits:
soft_limits = {
    'VaR Limit': {
        'example': 'Trader VaR = $10M',
        'rationale': 'Internal risk allocation',
        'override_reason': 'Exceptional opportunity',
        'approval': 'Desk head + risk manager'
    },
    
    'Position Limit': {
        'example': 'Max $100M in single stock',
        'rationale': 'Concentration risk',
        'override_reason': 'High-conviction trade',
        'approval': 'PM + CRO'
    },
    
    'Sector Limit': {
        'example': 'Max 30% in technology',
        'rationale': 'Diversification',
        'override_reason': 'Strong sector view',
        'approval': 'CIO'
    }
}
\`\`\`

**When to Use Each**

\`\`\`python
decision_framework = {
    'Use Hard Limit if:': [
        'External constraint (regulatory, contractual)',
        'Existential risk (could kill firm)',
        'No discretion allowed',
        'Binary yes/no decision',
        'Speed critical (no time for approval)'
    ],
    
    'Use Soft Limit if:': [
        'Internal risk management',
        'Judgment needed (context matters)',
        'Exceptional circumstances possible',
        'Risk vs opportunity tradeoff',
        'Senior oversight valuable'
    ]
}

# Example decisions:
examples = {
    'Regulatory position limit': 'HARD (no choice)',
    'Trader VaR limit': 'SOFT (internal allocation)',
    'Margin requirements': 'HARD (prime broker enforced)',
    'Single name concentration': 'SOFT (judgment call)',
    'Stop-loss at 10% down': 'SOFT (may want to hold)',
    'Max loss per day ($5M)': 'HARD (preserve capital)'
}
\`\`\`

**Override Procedures**

\`\`\`python
class LimitOverride:
    def request_override(self, trader, limit_type, amount, justification):
        """
        Formal override request workflow
        """
        request = {
            'timestamp': datetime.now(),
            'trader': trader.name,
            'limit_breached': limit_type,
            'current': self.get_current(trader, limit_type),
            'limit': self.get_limit(trader, limit_type),
            'requested': amount,
            'breach_pct': (amount - limit) / limit,
            'justification': justification,
            'status': 'PENDING'
        }
        
        # Determine approval chain
        approvers = self.get_approvers(limit_type, request['breach_pct'])
        
        # Send for approval
        for approver in approvers:
            self.notify_approver(approver, request)
        
        return request
    
    def get_approvers(self, limit_type, breach_pct):
        """
        Escalation based on severity
        """
        if breach_pct < 0.10:  # <10% breach
            return ['Desk Head']
        elif breach_pct < 0.25:  # 10-25% breach
            return ['Desk Head', 'Risk Manager']
        elif breach_pct < 0.50:  # 25-50% breach
            return ['Desk Head', 'Risk Manager', 'CRO']
        else:  # >50% breach
            return ['CRO', 'CFO', 'CEO']  # Senior management

# Override example
override_request = {
    'trader': 'Senior Trader',
    'limit': 'VaR $10M',
    'current': '$9.5M',
    'trade': 'Would increase to $12M',
    'breach': '20%',
    
    'justification': '''
        Exceptional opportunity in XYZ stock:
        - Earnings beat by 50%
        - Price down 10% on misunderstanding
        - Expected gain: $5M
        - Risk: $2M
        - Time-sensitive (closes in 1 hour)
    ''',
    
    'approvers': ['Desk Head', 'Risk Manager'],
    
    'approval_timeline': {
        '3:00pm': 'Request submitted',
        '3:05pm': 'Desk head approves (high conviction)',
        '3:15pm': 'Risk manager reviews',
        '3:20pm': 'Risk manager approves (conditional)',
        '3:21pm': 'Override granted - trade approved',
        'condition': 'Must reduce below limit by EOD'
    }
}
\`\`\`

**Override Tracking**
\`\`\`python
# All overrides logged and reviewed

override_analytics = {
    'frequency': 'How often each trader requests',
    'approval_rate': 'How often approved',
    'performance': 'P&L of override trades',
    'patterns': 'Are overrides clustered (risk)',
    
    'red_flags': {
        'frequent_requests': '>5 per month → misaligned limits',
        'always_approved': 'Limits too tight',
        'never_approved': 'Limits reasonable, trader pushing',
        'poor_performance': 'Override trades losing money → bad judgment'
    }
}

# Monthly review:
trader_override_report = {
    'trader_a': {
        'requests': 3,
        'approved': 3,
        'pnl': +5_000_000,  # Good judgment
        'action': 'Consider increasing limit'
    },
    
    'trader_b': {
        'requests': 12,
        'approved': 10,
        'pnl': -2_000_000,  # Bad judgment
        'action': 'Reduce limit, reject future overrides'
    }
}
\`\`\`

**Technology Implementation**
\`\`\`python
# Pre-trade compliance system

def execute_trade(trader, trade):
    # Check all limits
    hard_limit_check = check_hard_limits(trader, trade)
    if not hard_limit_check.passed:
        return {'status': 'REJECTED', 'reason': hard_limit_check.reason}
    
    soft_limit_check = check_soft_limits(trader, trade)
    if not soft_limit_check.passed:
        # Request override
        override_request = request_override(trader, trade, soft_limit_check)
        
        # Block until approval (or timeout)
        approval = wait_for_approval(override_request, timeout=300)  # 5 min
        
        if not approval.granted:
            return {'status': 'REJECTED', 'reason': 'Override denied'}
    
    # All limits passed/approved
    execute(trade)
    return {'status': 'EXECUTED'}

# Real-time (sub-second) limit checks
# Workflow system for override approvals (Bloomberg MSG, email, SMS)
\`\`\`

**Bottom Line**: Hard limits are absolute (regulatory, capacity, contractual). Soft limits allow judgment (internal risk management). Hard limits: auto-reject, no override. Soft limits: request approval, escalate based on severity. Override procedures: formal request, justification, approval chain, tracking. Good risk management: use hard limits for non-negotiable constraints, soft limits for internal risk allocations where context matters. Track override performance to calibrate limits.`,
    },
    {
      question:
        'Describe kill switches and circuit breakers in trading systems. What scenarios trigger these emergency stops, and how do firms balance automated protection with avoiding unnecessary trading halts?',
      answer: `Kill switches and circuit breakers are last-resort protections against catastrophic losses:

**Kill Switch**
\`\`\`python
class KillSwitch:
    """
    Emergency stop - halts ALL trading immediately
    
    Used when: System malfunction, rogue activity, catastrophic risk
    """
    
    def __init__(self):
        self.status = 'ACTIVE'  # or 'KILLED'
        self.triggers = self.load_kill_switch_triggers()
    
    def check_triggers(self):
        """
        Continuously monitor for kill switch conditions
        """
        for trigger in self.triggers:
            if trigger.condition_met():
                self.KILL(reason=trigger.reason)
    
    def KILL(self, reason):
        """
        IMMEDIATE ACTION:
        1. Block all new orders
        2. Cancel all pending orders
        3. Alert all traders
        4. Notify management
        5. Log for post-mortem
        """
        self.status = 'KILLED'
        self.block_all_new_orders()
        self.cancel_all_pending_orders()
        self.alert_emergency(reason)
        
        log.critical(f"KILL SWITCH ACTIVATED: {reason}")

# Kill switch triggers
kill_switch_triggers = {
    'Extreme Loss': {
        'threshold': '$50M loss in 10 minutes',
        'logic': 'Something catastrophically wrong',
        'action': 'KILL immediately'
    },
    
    'Abnormal Order Volume': {
        'threshold': '>10x normal order rate',
        'logic': 'System glitch or rogue algo',
        'action': 'KILL immediately'
    },
    
    'Position Explosion': {
        'threshold': 'Position >200% of limit',
        'logic': 'Controls failed',
        'action': 'KILL immediately'
    },
    
    'Market Dislocation': {
        'threshold': 'S&P moves >10% in 5 minutes',
        'logic': 'Market emergency (flash crash)',
        'action': 'KILL until assessed'
    },
    
    'Risk System Failure': {
        'threshold': 'Cannot calculate VaR for 5 minutes',
        'logic': 'Flying blind',
        'action': 'KILL until systems restored'
    }
}
\`\`\`

**Circuit Breakers**
\`\`\`python
class CircuitBreaker:
    """
    Graduated response - progressive trading restrictions
    
    Less drastic than kill switch
    Escalates based on severity
    """
    
    def __init__(self):
        self.level = 'NORMAL'  # NORMAL, YELLOW, RED
    
    def check_conditions(self):
        """
        Monitor for circuit breaker conditions
        """
        loss = self.get_current_loss()
        
        if loss > self.thresholds['red']:
            self.set_level('RED')
        elif loss > self.thresholds['yellow']:
            self.set_level('YELLOW')
        else:
            self.set_level('NORMAL')
    
    def set_level(self, level):
        """
        Escalating restrictions
        """
        if level == 'YELLOW':
            # Elevated risk - restrict
            self.actions = {
                'new_positions': 'Require approval',
                'position_increases': 'Blocked',
                'position_decreases': 'Allowed',
                'notification': 'Desk head + Risk'
            }
        
        elif level == 'RED':
            # Critical risk - aggressive restriction
            self.actions = {
                'new_positions': 'Blocked',
                'position_increases': 'Blocked',
                'position_decreases': 'Allowed (encouraged)',
                'forced_reduction': 'After 30 minutes',
                'notification': 'CRO + C-suite'
            }
        
        self.level = level
        self.execute_restrictions()

# Circuit breaker thresholds
circuit_breaker_thresholds = {
    'Trader Level': {
        'yellow': '$1M loss (50% of daily limit)',
        'red': '$2M loss (100% of daily limit)',
        'duration': 'Reset daily'
    },
    
    'Desk Level': {
        'yellow': '$10M loss',
        'red': '$20M loss',
        'duration': 'Reset daily'
    },
    
    'Firm Level': {
        'yellow': '$100M loss',
        'red': '$200M loss (capital at risk)',
        'duration': 'Reset daily'
    }
}
\`\`\`

**Real Example: Knight Capital (2012)**
\`\`\`python
# What happens without kill switches

knight_disaster = {
    'date': 'August 1, 2012',
    
    'incident': {
        'deployment': 'New trading software',
        'error': 'Old code activated (hadnt been deleted)',
        'behavior': 'Sent millions of erroneous orders',
        'duration': '45 minutes'
    },
    
    'timeline': {
        '9:30am': 'Market opens, algo starts',
        '9:31am': 'Unusual order activity',
        '9:35am': 'Traders notice problem',
        '9:40am': 'Unable to stop system quickly',
        '10:15am': 'Finally killed (45 min)',
        '10:30am': 'Count losses: $440M'
    },
    
    'why_so_bad': {
        'no_kill_switch': 'No immediate emergency stop',
        'manual_process': 'Had to manually identify and stop',
        'complex_architecture': 'Multiple servers, hard to kill all',
        'no_automated_detection': 'Didnt detect abnormal behavior'
    },
    
    'outcome': 'Company nearly bankrupt, sold for scraps'
}

# Lesson: Need automated kill switches that trigger INSTANTLY
\`\`\`

**Balancing Protection vs False Alarms**
\`\`\`python
# Challenge: Too sensitive = unnecessary halts, too loose = disasters

calibration_tradeoff = {
    'Too Sensitive': {
        'problem': 'Kill switch triggers on normal volatility',
        'cost': 'Lost trading opportunities',
        'example': '$5M loss threshold → triggers 5x per year',
        'consequence': 'Traders override or disable (defeats purpose)'
    },
    
    'Too Loose': {
        'problem': 'Doesnt trigger until catastrophic',
        'cost': 'Large losses before stop',
        'example': '$100M loss threshold → never triggers until disaster',
        'consequence': 'Knight Capital scenario'
    },
    
    'Right Balance': {
        'approach': 'Multiple thresholds + human judgment',
        'structure': {
            'automated_kill': 'Only for extreme/abnormal',
            'circuit_breakers': 'Graduated response for losses',
            'manual_override': 'Humans can kill anytime',
            'testing': 'Regular drills (monthly)'
        }
    }
}
\`\`\`

**Implementation Best Practices**
\`\`\`python
kill_switch_implementation = {
    '1. Multiple Independent Triggers': {
        'loss_based': '$50M in 10 min',
        'volume_based': '10x normal order rate',
        'position_based': '200% of limit',
        'system_based': 'Risk system down',
        'manual': 'Any senior trader can hit button'
    },
    
    '2. Physical Kill Switch': {
        'location': 'Trading floor - big red button',
        'access': 'Anyone can press',
        'action': 'Immediate halt',
        'reason': 'In emergency, dont wait for algorithm'
    },
    
    '3. Automated Detection': {
        'ml_model': 'Detect abnormal trading patterns',
        'anomaly_detection': 'Statistical outliers',
        'peer_comparison': 'This trader vs others',
        'historical': 'Today vs typical day'
    },
    
    '4. Graduated Response': {
        'level_1': 'Alert + require approval',
        'level_2': 'Block new risk',
        'level_3': 'Force reduction',
        'level_4': 'KILL ALL'
    },
    
    '5. Post-Mortem Process': {
        'every_trigger': 'Review why triggered',
        'false_positive': 'Adjust thresholds',
        'true_positive': 'Prevented disaster - good',
        'quarterly_review': 'Calibrate thresholds'
    }
}
\`\`\`

**Testing and Drills**
\`\`\`python
# Must test regularly (like fire drills)

kill_switch_drill = {
    'frequency': 'Monthly',
    
    'scenario': 'Simulated rogue algo',
    
    'procedure': [
        '1. Risk manager triggers kill switch',
        '2. Verify all order entry blocked',
        '3. Verify pending orders cancelled',
        '4. Verify alerts sent',
        '5. Time to full stop (target: <10 seconds)',
        '6. Restore procedures (bring systems back)',
        '7. Document lessons learned'
    ],
    
    'metrics': {
        'response_time': 'How fast to full stop?',
        'completeness': 'All systems stopped?',
        'recovery_time': 'How fast to restore?',
        'communication': 'Everyone notified?'
    }
}

# Regulatory requirement: Annual testing minimum
# Best practice: Monthly
\`\`\`

**Bottom Line**: Kill switches halt ALL trading for extreme scenarios (catastrophic loss, system malfunction, rogue activity). Circuit breakers provide graduated response to escalating losses (yellow = restrict, red = near-halt). Knight Capital disaster shows necessity of automated, instant kill switches. Challenge: balance protection vs false alarms. Solution: multiple independent triggers, graduated response, human override, regular testing. Physical red button on trading floor for emergencies. Test monthly to ensure works when needed.`,
    },
  ],
} as const;
