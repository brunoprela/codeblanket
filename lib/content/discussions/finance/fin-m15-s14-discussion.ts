export default {
  id: 'fin-m15-s14-discussion',
  title: 'Risk Reporting and Dashboards - Discussion Questions',
  questions: [
    {
      question:
        'Explain the different risk reporting requirements for various stakeholders (board, regulators, senior management, desk heads). How do reporting frequency, detail level, and format differ across audiences?',
      answer: `Risk reporting must be tailored to each audience's needs and regulatory requirements:

**Stakeholder Requirements**
\`\`\`python
reporting_requirements = {
    'Board of Directors': {
        'frequency': 'Quarterly (monthly for large firms)',
        'detail': 'High-level summary',
        'metrics': [
            'Firm-wide VaR trend',
            'VaR vs limit utilization',
            'Stress test results',
            'Top 10 risks',
            'Limit breaches (material)',
            'Regulatory capital'
        ],
        'format': 'Executive summary (1-2 pages) + appendix',
        'focus': 'Is firm within risk appetite?',
        'time_to_prepare': '1-2 weeks'
    },
    
    'Regulators': {
        'frequency': 'Daily (some metrics), Monthly/Quarterly (comprehensive)',
        'detail': 'Very detailed, standardized format',
        'reports': {
            'daily': 'VaR, Stress VaR, backtesting',
            'monthly': 'FR Y-9C (US), COREP (EU)',
            'quarterly': 'Capital adequacy, liquidity',
            'annual': 'Stress test (CCAR/DFAST)'
        },
        'format': 'Regulatory templates (fixed format)',
        'penalties': 'Fines for late/incorrect reporting',
        'time_to_prepare': 'Ongoing + month-end close'
    },
    
    'CEO/CFO/CRO': {
        'frequency': 'Daily brief + weekly detailed',
        'detail': 'Medium (can drill down)',
        'metrics': [
            'P&L by business unit',
            'VaR by desk',
            'Top exposures',
            'Limit breaches',
            'Market intelligence'
        ],
        'format': 'Dashboard + 1-page summary',
        'focus': 'What needs attention?',
        'time_to_prepare': 'Automated (EOD)'
    },
    
    'Desk Heads': {
        'frequency': 'Real-time + daily detailed',
        'detail': 'Very detailed (position-level)',
        'metrics': [
            'Desk P&L',
            'Trader P&L breakdown',
            'VaR by trader',
            'Limit utilization',
            'Attribution analysis'
        ],
        'format': 'Interactive dashboard',
        'focus': 'Manage traders, optimize allocation',
        'time_to_prepare': 'Real-time'
    },
    
    'Traders': {
        'frequency': 'Real-time',
        'detail': 'Position-level',
        'metrics': [
            'My P&L',
            'My VaR vs limit',
            'My positions',
            'Greeks'
        ],
        'format': 'Real-time dashboard',
        'focus': 'Stay within limits, optimize trades',
        'time_to_prepare': 'Real-time'
    }
}
\`\`\`

**Report Detail by Level**
\`\`\`python
# Example: VaR reporting

board_var_report = {
    'content': 'Single number: Firm VaR = $150M',
    'trend': 'Chart showing last 12 months',
    'commentary': '2 sentences on why it changed',
    'page_count': '1 page'
}

cro_var_report = {
    'content': 'VaR by business unit (5 units)',
    'breakdown': 'Equity, Fixed Income, etc.',
    'drivers': 'Top 5 risk factors',
    'commentary': '1 paragraph per unit',
    'page_count': '5 pages'
}

desk_head_var_report = {
    'content': 'VaR by trader (20 traders)',
    'breakdown': 'Position-level contribution',
    'attribution': 'Factor-based risk attribution',
    'commentary': 'Detailed notes on each trader',
    'page_count': '20 pages + appendix'
}

# Pyramid: Detail increases as you go down
\`\`\`

**Regulatory Reporting Examples**

**Basel Regulatory Capital**
\`\`\`python
# Basel Pillar 3 disclosure (quarterly)

basel_pillar_3 = {
    'Market Risk': {
        'var': 'Trading book VaR',
        'stressed_var': 'Stressed VaR',
        'irc': 'Incremental Risk Charge',
        'crc': 'Comprehensive Risk Capital',
        'total_capital': 'Sum × multiplier (3+)'
    },
    
    'Credit Risk': {
        'exposure': 'Credit exposures by counterparty type',
        'rwa': 'Risk-Weighted Assets',
        'capital': 'Required capital (8% of RWA)'
    },
    
    'Operational Risk': {
        'approach': 'Standardized or AMA',
        'capital': 'Op risk capital requirement'
    },
    
    'format': 'Standardized tables + narrative',
    'public': True,  # Published on website
    'deadline': '90 days after quarter end'
}
\`\`\`

**CCAR/DFAST (US Stress Testing)**
\`\`\`python
# Annual stress test submission to Fed

ccar_submission = {
    'scenarios': [
        'Baseline',
        'Adverse',
        'Severely Adverse (Fed scenario)',
        'Company-designed scenarios (2+)'
    ],
    
    'projections': {
        'horizon': '9 quarters',
        'variables': [
            'Net income',
            'Losses by portfolio',
            'Pre-provision net revenue',
            'Loan loss provisions',
            'Capital ratios',
            'RWA'
        ]
    },
    
    'detail': {
        'models': 'Documentation of all models',
        'assumptions': 'Economic assumptions',
        'methodology': 'How stress applied',
        'validation': 'Independent validation'
    },
    
    'submission': 'April 5 deadline',
    'effort': '500-1000 person-hours',
    'results': 'Fed publishes in June'
}
\`\`\`

**Report Automation**
\`\`\`python
class RiskReportingSystem:
    """
    Automated risk reporting pipeline
    """
    
    def generate_daily_reports(self):
        """
        EOD automated reports
        """
        # Extract data
        data = {
            'var': self.calculate_firm_var(),
            'pnl': self.get_pnl_by_desk(),
            'limits': self.check_limit_breaches(),
            'concentrations': self.identify_concentrations()
        }
        
        # Generate reports for each audience
        reports = {
            'cro_brief': self.format_cro_brief(data),
            'desk_reports': self.format_desk_reports(data),
            'regulatory': self.format_regulatory_reports(data)
        }
        
        # Distribute
        self.email_report(reports['cro_brief'], recipients=['CRO', 'CFO'])
        self.publish_to_dashboard(reports['desk_reports'])
        self.submit_to_regulator(reports['regulatory'])
        
        return reports
    
    def generate_board_report(self, quarter):
        """
        Quarterly board report (manual + automated)
        """
        # Automated data extraction
        data = self.extract_quarter_data(quarter)
        
        # Template-based report generation
        template = self.load_board_template()
        
        # Fill template
        report = template.render(data)
        
        # Manual review and commentary
        report = self.add_manual_commentary(report)
        
        # Format (PowerPoint)
        presentation = self.create_presentation(report)
        
        return presentation

# Time saved: 80% automation
# From 40 hours/week to 8 hours/week
\`\`\`

**Best Practices**

\`\`\`python
reporting_best_practices = {
    'Timeliness': {
        'real_time': 'Traders, desk heads',
        'eod': 'Senior management',
        'next_day': 'Never acceptable (outdated)',
        'tip': 'Automate everything possible'
    },
    
    'Accuracy': {
        'validation': 'Automated checks before distribution',
        'reconciliation': 'Cross-check against systems',
        'review': 'Risk manager sign-off',
        'tip': 'Never sacrifice accuracy for speed'
    },
    
    'Actionability': {
        'exceptions': 'Highlight what needs attention',
        'context': 'Why did it change?',
        'recommendations': 'What should we do?',
        'tip': 'Dont just report numbers, add insight'
    },
    
    'Consistency': {
        'format': 'Same format each time',
        'definitions': 'Consistent metric definitions',
        'comparability': 'Show trends, prior periods',
        'tip': 'Stakeholders want consistency'
    },
    
    'Visualization': {
        'board': 'High-level charts (few numbers)',
        'management': 'Balanced (charts + tables)',
        'analysts': 'Detailed tables',
        'tip': 'Match viz to audience sophistication'
    }
}
\`\`\`

**Bottom Line**: Risk reporting must be tailored to audience. Board: high-level, quarterly, focus on appetite. Regulators: detailed, standardized, frequent (daily/monthly). Management: dashboard, daily/weekly, actionable. Desk heads: detailed, real-time, position-level. Traders: real-time only their positions. Detail pyramid: more detail as you go down. Automate 80%+ (data extraction, report generation, distribution). Manual effort: commentary, analysis, insight. Key: timeliness, accuracy, actionability, consistency, appropriate visualization.`,
    },
    {
      question:
        'Describe effective risk report visualization techniques. When should you use heat maps, trend charts, gauges, or tables, and what makes a risk dashboard actionable rather than just informative?',
      answer: `Effective visualization transforms data into insight and action:

**Visualization by Purpose**
\`\`\`python
visualization_guide = {
    'Heat Map': {
        'use_when': 'Showing many items at once, scanning for problems',
        'best_for': [
            'Desk × Risk Metric grid',
            'Trader limit utilization',
            'Correlation matrices',
            'Position concentrations'
        ],
        'example': '''
                  VaR    P&L   Limit%
        Equity    🟢     🟢     🟡
        FI        🟡     🔴     🟢
        Deriv     🟢     🟢     🟢
        ''',
        'advantage': 'Instant pattern recognition',
        'limitation': 'Loses precise numbers'
    },
    
    'Trend Chart': {
        'use_when': 'Showing change over time',
        'best_for': [
            'P&L over day/month/year',
            'VaR trend',
            'Limit utilization history',
            'Stress test results'
        ],
        'advantage': 'Shows momentum, seasonality',
        'limitation': 'Takes space, only 1-3 series readable'
    },
    
    'Gauge/Speedometer': {
        'use_when': 'Showing single metric vs target',
        'best_for': [
            'VaR vs limit',
            'LCR vs 100%',
            'Capital ratio vs requirement'
        ],
        'advantage': 'Instant red/yellow/green',
        'limitation': 'Only for single number'
    },
    
    'Table': {
        'use_when': 'Precise numbers needed, drill-down',
        'best_for': [
            'Position list',
            'Trade blotter',
            'Detailed attribution'
        ],
        'advantage': 'Precise, sortable, filterable',
        'limitation': 'Hard to scan, cognitive load'
    },
    
    'Bar Chart': {
        'use_when': 'Comparing discrete items',
        'best_for': [
            'P&L by desk',
            'VaR contribution by asset class',
            'Top 10 positions'
        ],
        'advantage': 'Easy comparison',
        'limitation': 'Max ~20 bars before unreadable'
    }
}
\`\`\`

**Actionable Dashboard Design**

**Example: Trading Desk Risk Dashboard**
\`\`\`python
actionable_dashboard = {
    'Top Bar (Status)': {
        'visual': 'Red/Yellow/Green indicator',
        'logic': {
            'green': 'All limits OK, P&L positive',
            'yellow': 'Approaching limits or small loss',
            'red': 'Breach or large loss'
        },
        'action': 'Red = immediate attention needed',
        'size': 'Full width banner'
    },
    
    'Primary Metrics (Large, Prominent)': {
        'pnl_today': {
            'visual': 'Large number, color-coded',
            'example': '+$2.5M (green) or -$1.2M (red)',
            'action_trigger': 'If red >$2M, investigate',
            'drill_down': 'Click → position-level attribution'
        },
        
        'var_gauge': {
            'visual': 'Speedometer',
            'ranges': '0-70% green, 70-90% yellow, 90-100% orange, >100% red',
            'action_trigger': 'If yellow, plan to reduce',
            'drill_down': 'Click → risk factor breakdown'
        },
        
        'limit_breaches': {
            'visual': 'Count badge (like notifications)',
            'example': '3 breaches',
            'action_trigger': 'Any breach requires action',
            'drill_down': 'Click → list of breaches'
        }
    },
    
    'Secondary Metrics (Scannable)': {
        'trader_heat_map': {
            'visual': 'Grid of trader names, colored by status',
            'example': '''
            Trader A: 🟢  Trader B: 🟡  Trader C: 🔴
            Trader D: 🟢  Trader E: 🟢  Trader F: 🟡
            ''',
            'action_trigger': 'Red = talk to trader',
            'drill_down': 'Click → trader detail'
        },
        
        'pnl_trend': {
            'visual': 'Intraday line chart',
            'overlay': 'Previous day (gray) vs today (blue)',
            'action_trigger': 'Unusual pattern → investigate',
            'time_range': 'Market open to now'
        },
        
        'top_risks': {
            'visual': 'Bar chart of risk contributions',
            'example': 'Interest rates: 40%, Credit: 30%, Equity: 20%',
            'action_trigger': 'Concentration >50% in one factor',
            'drill_down': 'Click → hedging opportunities'
        }
    },
    
    'Alerts Section': {
        'visual': 'List with icons',
        'prioritized': 'Critical (red) at top',
        'example': '''
        🔴 Trader C: VaR breach 120%
        🟡 Interest rate risk >50%
        🔵 New trade booked: $50M
        ''',
        'action_trigger': 'Each alert has action button',
        'interactions': 'Acknowledge, Snooze, Take Action'
    }
}

# Key principle: Top-left = most important
# Visual hierarchy: Size + position = importance
# Color: Sparingly (red = danger only)
\`\`\`

**Actionable vs Informative**
\`\`\`python
comparison = {
    'Informative Only': {
        'shows': 'VaR = $52M',
        'problem': 'So what? Is that good or bad?',
        'user_thought': 'Interesting... I guess?'
    },
    
    'Actionable': {
        'shows': 'VaR = $52M / $50M limit (104% - BREACH)',
        'implication': 'Over limit!',
        'action': 'Button: "Reduce VaR" → shows positions to cut',
        'user_thought': 'I need to act now'
    },
    
    'Informative Only': {
        'shows': 'P&L = -$2.5M',
        'problem': 'Why? What happened?',
        'user_thought': 'Need to investigate...'
    },
    
    'Actionable': {
        'shows': 'P&L = -$2.5M',
        'attribution': {
            'rates_up': '-$3M (Fed hike)',
            'spreads_tight': '+$500K',
            'unexplained': '$0 ✓'
        },
        'action': 'Button: "Hedge rates" → suggested trades',
        'user_thought': 'Ah, rates hurt. Should I hedge?'
    }
}

# Actionable = Information + Context + Next Steps
\`\`\`

**Interactive Features**
\`\`\`python
interactive_elements = {
    'Drill-Down': {
        'hierarchy': 'Firm → Desk → Trader → Position',
        'implementation': 'Click any number to see breakdown',
        'benefit': 'Start high-level, zoom to detail as needed'
    },
    
    'Filtering': {
        'options': [
            'Show only breaches',
            'Hide traders with P&L > $0',
            'Only show large positions (>$10M)'
        ],
        'benefit': 'Focus on what matters to you',
        'reset': 'One-click to reset all filters'
    },
    
    'Time Period': {
        'options': 'Today, MTD, YTD, Custom',
        'comparison': 'Compare to prior period',
        'benefit': 'Context for current metrics'
    },
    
    'Scenario Analysis': {
        'feature': '"What if" calculator',
        'example': 'If I cut position X by 50%, VaR would be...',
        'benefit': 'Test actions before executing',
        'implementation': 'Real-time recalculation'
    },
    
    'Action Buttons': {
        'examples': [
            '"Reduce VaR" → list of positions to cut',
            '"Hedge rates" → suggested IR swaps',
            '"View details" → position-level breakdown'
        ],
        'benefit': 'From insight to action in one click'
    }
}
\`\`\`

**Common Mistakes to Avoid**
\`\`\`python
dashboard_mistakes = {
    'Too Much Data': {
        'problem': '50 metrics on one screen',
        'result': 'Cognitive overload, nothing stands out',
        'fix': 'Max 8-10 key metrics, rest in drill-down'
    },
    
    'No Context': {
        'problem': 'Shows VaR = $52M (is that normal?)',
        'result': 'User doesnt know if action needed',
        'fix': 'Always show: vs limit, vs yesterday, trend'
    },
    
    'Stale Data': {
        'problem': 'Last updated: 2 hours ago',
        'result': 'User doesnt trust it',
        'fix': 'Real-time updates + timestamp'
    },
    
    'Wrong Visualizations': {
        'problem': 'Pie chart with 20 slices',
        'result': 'Unreadable',
        'fix': 'Pie charts only for 2-5 slices, else use bar'
    },
    
    'No Prioritization': {
        'problem': 'All metrics same size/color',
        'result': 'What should I look at first?',
        'fix': 'Visual hierarchy: size + position + color'
    },
    
    'Death by Numbers': {
        'problem': 'Every metric to 6 decimal places',
        'result': 'Cant see the forest for trees',
        'fix': 'Round aggressively ($52.3M not $52,347,182.19)'
    }
}
\`\`\`

**Mobile Considerations**
\`\`\`python
# Risk managers need mobile access

mobile_dashboard = {
    'priority_1': 'Status indicator (red/yellow/green)',
    'priority_2': 'Firm P&L + VaR utilization',
    'priority_3': 'Active alerts (count)',
    'drill_down': 'Tap for details',
    
    'omit': [
        'Complex charts (unreadable on phone)',
        'Large tables (too much scrolling)',
        'Fine details (save for desktop)'
    ],
    
    'mobile_specific': {
        'notifications': 'Push alerts for critical issues',
        'quick_actions': 'Approve/reject with one tap',
        'offline': 'Cache last data for offline view'
    }
}
\`\`\`

**Bottom Line**: Effective visualization matches viz type to purpose: heat maps for scanning, gauges for single metrics, trends for time series, tables for detail. Actionable dashboard = information + context + next steps. Design hierarchy: top-left most important, size = importance, color sparingly (red = danger). Interactive: drill-down, filters, scenario analysis, action buttons. Avoid: too much data, no context, stale data, wrong viz type, no prioritization. Mobile: prioritize status, P&L, alerts; omit complexity. Goal: user sees problem → understands why → knows what to do → can act immediately.`,
    },
    {
      question:
        'Explain the difference between regular risk reports and exception reports. How should firms design exception-based reporting to surface issues without creating alert fatigue?',
      answer: `Exception-based reporting focuses scarce attention on what truly matters:

**Regular vs Exception Reports**
\`\`\`python
comparison = {
    'Regular Report': {
        'content': 'All metrics, every time',
        'example': '100-page PDF with every desk, trader, position',
        'frequency': 'Daily/weekly',
        'reader_effort': 'High (scan 100 pages for issues)',
        'problem': 'Needle in haystack',
        'time_to_find_issue': '10-30 minutes'
    },
    
    'Exception Report': {
        'content': 'Only items requiring attention',
        'example': '1-page: 3 limit breaches, 2 large losses',
        'frequency': 'As needed + daily summary',
        'reader_effort': 'Low (all items need action)',
        'advantage': 'Focuses attention',
        'time_to_find_issue': '10 seconds'
    }
}

# Principle: "No news is good news"
# Only report exceptions
\`\`\`

**Exception Definition**
\`\`\`python
class ExceptionDetector:
    """
    What qualifies as an exception?
    """
    
    def __init__(self):
        self.thresholds = self.load_exception_thresholds()
    
    def is_exception(self, metric, value):
        """
        Rules for exception detection
        """
        exceptions = []
        
        # 1. Limit breaches
        if value > metric.limit:
            exceptions.append({
                'type': 'BREACH',
                'severity': 'HIGH',
                'message': f'{metric.name} breached: {value} > {metric.limit}'
            })
        
        # 2. Large moves
        change = abs(value - metric.previous_value)
        if change > metric.move_threshold:
            exceptions.append({
                'type': 'LARGE_MOVE',
                'severity': 'MEDIUM',
                'message': f'{metric.name} moved {change} (>{metric.move_threshold})'
            })
        
        # 3. Unusual patterns
        if self.is_statistical_outlier(metric, value):
            exceptions.append({
                'type': 'OUTLIER',
                'severity': 'MEDIUM',
                'message': f'{metric.name} is statistical outlier'
            })
        
        # 4. Approaching limits
        utilization = value / metric.limit
        if 0.8 < utilization < 1.0:
            exceptions.append({
                'type': 'WARNING',
                'severity': 'LOW',
                'message': f'{metric.name} at {utilization:.0%} of limit'
            })
        
        return exceptions

# Example exception thresholds:
exception_thresholds = {
    'var': {
        'breach': 'VaR > limit',
        'warning': 'VaR > 80% of limit',
        'large_move': 'VaR change >20% day-over-day'
    },
    
    'pnl': {
        'large_loss': 'P&L < -$5M',
        'large_gain': 'P&L > +$10M (suspicious?)',
        'large_move': 'P&L change >$5M intraday'
    },
    
    'concentration': {
        'single_name': 'Position >10% of portfolio',
        'sector': 'Sector >30% of portfolio'
    }
}
\`\`\`

**Exception Report Structure**
\`\`\`python
exception_report = {
    'Header (Summary)': {
        'total_exceptions': '5',
        'critical': '1',
        'important': '3',
        'informational': '1',
        'color_code': 'RED (has critical)'
    },
    
    'Critical Section (Must Act)': [
        {
            'desk': 'Fixed Income',
            'trader': 'Trader A',
            'issue': 'VaR breach',
            'value': '$12M',
            'limit': '$10M',
            'breach_pct': '20%',
            'action_required': 'Reduce position by EOD',
            'owner': 'Desk Head',
            'status': 'In Progress'
        }
    ],
    
    'Important Section (Should Act Soon)': [
        {
            'desk': 'Equity',
            'issue': 'Large loss',
            'value': '-$6M',
            'reason': 'Tech sector down 5%',
            'action_suggested': 'Consider hedge',
            'owner': 'Desk Head',
            'status': 'Monitoring'
        },
        # ... 2 more
    ],
    
    'Informational (FYI)': [
        {
            'desk': 'Derivatives',
            'issue': 'VaR approaching limit',
            'value': '$8.5M / $10M (85%)',
            'action': 'Monitor',
            'owner': 'Risk Manager'
        }
    ],
    
    'Footer': {
        'full_report': 'Link to complete daily report',
        'archived': 'Link to historical exceptions'
    }
}

# Key: Prioritized, actionable, owner assigned
\`\`\`

**Preventing Alert Fatigue**
\`\`\`python
alert_fatigue_prevention = {
    '1. Prioritization': {
        'problem': 'All alerts treated equally',
        'solution': 'Critical / Important / Info tiers',
        'implementation': {
            'critical': 'Pop-up + SMS + email',
            'important': 'Email + dashboard',
            'info': 'Dashboard only'
        },
        'benefit': 'Users know what needs immediate attention'
    },
    
    '2. Deduplication': {
        'problem': 'Same alert 50 times',
        'solution': 'Group related alerts',
        'example': '''
            BAD:  50 alerts "Position X over limit"
            GOOD: 1 alert "Position X over limit (50 instances today)"
        ''',
        'benefit': 'Reduce noise'
    },
    
    '3. Smart Thresholds': {
        'problem': 'Alert triggers on normal volatility',
        'solution': 'Dynamic thresholds based on volatility regime',
        'example': {
            'low_vol_period': 'Alert if VaR >10% increase',
            'high_vol_period': 'Alert if VaR >30% increase'
        },
        'benefit': 'Fewer false positives'
    },
    
    '4. Escalation': {
        'problem': 'Critical alert ignored',
        'solution': 'Auto-escalate if not acknowledged',
        'timeline': {
            '0 min': 'Alert to owner',
            '5 min': 'If not acknowledged, alert manager',
            '10 min': 'If not acknowledged, alert CRO + phone call'
        },
        'benefit': 'Ensures critical issues addressed'
    },
    
    '5. Context': {
        'problem': 'Alert with no context',
        'solution': 'Include why it matters + what to do',
        'example_bad': 'VaR = $12M',
        'example_good': '''
            VaR = $12M (20% over $10M limit)
            Reason: Large new position in Tech
            Action: Reduce Tech by $50M or hedge
            Deadline: EOD
        ''',
        'benefit': 'Actionable, not just informative'
    },
    
    '6. Feedback Loop': {
        'problem': 'Alerts not tuned over time',
        'solution': 'Track false positive rate',
        'metrics': {
            'true_positive': 'Alert → action taken',
            'false_positive': 'Alert → no action (not real issue)',
            'target': '<10% false positive rate'
        },
        'action': 'Adjust thresholds quarterly',
        'benefit': 'Continuously improve alert quality'
    }
}
\`\`\`

**Example Exception-Based Workflow**
\`\`\`python
daily_exception_workflow = {
    '7:00 AM - EOD Processing': {
        'system': 'Run exception detection on EOD data',
        'detected': '25 potential exceptions'
    },
    
    '7:15 AM - Filtering': {
        'system': 'Apply deduplication, grouping',
        'filtered_to': '8 unique exceptions'
    },
    
    '7:20 AM - Prioritization': {
        'system': 'Classify by severity',
        'critical': 1,
        'important': 4,
        'info': 3
    },
    
    '7:30 AM - Assignment': {
        'system': 'Assign owner based on rules',
        'owners': {
            'critical': 'CRO',
            'important': 'Desk heads',
            'info': 'Risk managers'
        }
    },
    
    '7:35 AM - Distribution': {
        'system': 'Send exception report',
        'channels': {
            'critical': 'SMS + email + Bloomberg',
            'important': 'Email',
            'info': 'Dashboard'
        }
    },
    
    '8:00 AM - Review': {
        'users': 'Review exceptions',
        'time_required': '5 minutes (vs 30 min for full report)',
        'actions': 'Acknowledge, assign actions'
    },
    
    'Intraday - Real-Time': {
        'system': 'Monitor for new exceptions',
        'alert_immediately': 'Critical exceptions',
        'batch': 'Non-critical (hourly summary)'
    }
}
\`\`\`

**Machine Learning for Exception Detection**
\`\`\`python
# Advanced: ML to detect anomalies

class MLExceptionDetector:
    """
    Use ML to find unusual patterns
    """
    
    def train(self, historical_data):
        """
        Learn normal patterns from history
        """
        # Train anomaly detection model
        self.model = IsolationForest()
        self.model.fit(historical_data)
    
    def detect_exceptions(self, current_data):
        """
        Flag statistical outliers
        """
        # Predict: -1 = anomaly, 1 = normal
        predictions = self.model.predict(current_data)
        
        exceptions = []
        for i, pred in enumerate(predictions):
            if pred == -1:
                exceptions.append({
                    'item': current_data[i],
                    'reason': 'Statistical anomaly',
                    'severity': 'MEDIUM'
                })
        
        return exceptions

# Use cases:
ml_exception_use_cases = {
    'Unusual P&L pattern': 'Detect P&L that doesnt fit historical',
    'Correlation breaks': 'Assets moving differently than expected',
    'Trading patterns': 'Trader behavior anomaly',
    'Risk factor shifts': 'Sudden change in risk composition'
}

# Benefit: Catch issues that fixed thresholds miss
# Challenge: Higher false positive rate initially
\`\`\`

**Bottom Line**: Exception reports show only what needs attention (breaches, large moves, outliers, warnings). Much more efficient than full reports (5 min vs 30 min). Prevent alert fatigue: prioritize (critical/important/info), deduplicate, smart dynamic thresholds, escalate if ignored, provide context, tune based on false positive rate. Workflow: EOD detection → filter → prioritize → assign owner → distribute. Real-time for critical, batched for non-critical. ML can detect statistical anomalies fixed thresholds miss. Goal: "No news is good news"—only report exceptions.`,
    },
  ],
} as const;
