import { Content } from '@/lib/types';

const paperTradingVsLiveDiscussion: Content = {
    title: 'Paper Trading vs Live Trading - Discussion Questions',
    description:
        'Deep-dive discussion questions on transition strategies, risk management, and production deployment',
    sections: [
        {
            title: 'Discussion Questions',
            content: `
# Discussion Questions: Paper Trading vs Live Trading

## Question 1: Paper Trading Performance Divergence Analysis

**Scenario**: After 6 months of paper trading, your momentum strategy shows concerning differences from backtesting:

**Backtesting Results (2018-2023)**:
- Sharpe Ratio: 2.1
- Max Drawdown: -12%
- Win Rate: 58%
- Avg Trade Duration: 3.2 days

**Paper Trading Results (Last 6 months)**:
- Sharpe Ratio: 0.9
- Max Drawdown: -22%
- Win Rate: 52%
- Avg Trade Duration: 4.1 days
- Slippage averaging 15 bps (backtest assumed 5 bps)
- Several limit orders went unfilled (backtest assumed all fills)

**Questions:**
1. What factors explain the performance degradation?
2. Should you proceed to live trading, modify the strategy, or abandon it?
3. How do you adjust backtest assumptions for future strategies?

### Comprehensive Answer

This scenario illustrates typical paper trading revelations. Let's analyze systematically:

**Factor Analysis:**

\`\`\`python
class PaperTradingPostMortem:
    """
    Analyze paper trading vs backtest divergence
    """
    
    def __init__(
        self,
        backtest_results: Dict,
        paper_results: Dict
    ):
        self.bt = backtest_results
        self.pt = paper_results
    
    def analyze_degradation(self) -> Dict:
        """Comprehensive degradation analysis"""
        
        analysis = {}
        
        # 1. Sharpe degradation
        sharpe_drop = (self.bt['sharpe'] - self.pt['sharpe']) / self.bt['sharpe']
        analysis['sharpe_degradation_pct'] = sharpe_drop * 100
        
        # Interpretation
        if sharpe_drop > 0.5:
            analysis['sharpe_severity'] = "SEVERE"
            analysis['sharpe_interpretation'] = (
                "57% Sharpe degradation indicates major issues. "
                "Backtest assumptions were unrealistic."
            )
        elif sharpe_drop > 0.3:
            analysis['sharpe_severity'] = "MODERATE"
            analysis['sharpe_interpretation'] = "Expected degradation range."
        else:
            analysis['sharpe_severity'] = "MINIMAL"
        
        # 2. Slippage impact
        backtest_slippage_bps = 5
        actual_slippage_bps = 15
        slippage_diff = actual_slippage_bps - backtest_slippage_bps
        
        # Estimate: If strategy trades 50 times with avg 10bps excess slippage
        estimated_slippage_cost = 50 * 0.0010  # 50 trades × 10 bps = 5% annual drag
        
        analysis['slippage'] = {
            'estimated_annual_cost_pct': estimated_slippage_cost * 100,
            'explanation': (
                f"Slippage 3x higher than assumed ({actual_slippage_bps} vs {backtest_slippage_bps} bps). "
                f"For 50 trades/year, this adds ~{estimated_slippage_cost*100:.1f}% cost."
            )
        }
        
        # 3. Fill rate impact
        backtest_fill_rate = 1.0
        paper_fill_rate = 0.92  # Estimate from "several unfilled"
        missed_trades_pct = (1 - paper_fill_rate) * 100
        
        analysis['fill_rate'] = {
            'missed_trades_pct': missed_trades_pct,
            'impact': (
                f"{missed_trades_pct:.0f}% of trades unfilled. "
                "Missed opportunities reduce returns, increase variance."
            )
        }
        
        # 4. Holding period extension
        bt_duration = 3.2
        pt_duration = 4.1
        duration_increase = (pt_duration - bt_duration) / bt_duration * 100
        
        analysis['holding_period'] = {
            'increase_pct': duration_increase,
            'explanation': (
                f"Trades held {duration_increase:.0f}% longer. "
                "Likely due to limit orders waiting for fills, reducing turnover."
            )
        }
        
        # 5. Max drawdown increase
        dd_increase = (self.pt['max_dd'] - self.bt['max_dd']) / abs(self.bt['max_dd']) * 100
        
        analysis['drawdown'] = {
            'increase_pct': dd_increase,
            'severity': "HIGH" if dd_increase > 50 else "MODERATE",
            'explanation': (
                f"Drawdown {dd_increase:.0f}% worse than backtest. "
                "Risk management may need adjustment."
            )
        }
        
        return analysis
    
    def make_recommendation(self, analysis: Dict) -> Dict:
        """Generate go/no-go recommendation"""
        
        # Scoring system
        score = 100
        
        # Penalize based on factors
        if analysis['sharpe_degradation_pct'] > 50:
            score -= 40  # Major penalty
        elif analysis['sharpe_degradation_pct'] > 30:
            score -= 20
        
        if analysis['slippage']['estimated_annual_cost_pct'] > 3:
            score -= 20
        
        if analysis['fill_rate']['missed_trades_pct'] > 10:
            score -= 15
        
        if analysis['drawdown']['increase_pct'] > 50:
            score -= 15
        
        # Make decision
        if score >= 70:
            decision = "PROCEED"
            recommendation = (
                "Proceed to live trading with reduced capital (10-20% of target). "
                "Performance degradation within acceptable range."
            )
        elif score >= 50:
            decision = "CONDITIONAL"
            recommendation = (
                "Conditional approval. Modify strategy to address execution issues. "
                "Consider: wider profit targets, longer holding periods, "
                "or different order types. Re-paper-trade for 2 months after changes."
            )
        else:
            decision = "REJECT"
            recommendation = (
                "Do not proceed to live trading. Degradation too severe. "
                "Strategy not robust to real-world conditions. "
                "Options: major redesign or abandon."
            )
        
        return {
            'score': score,
            'decision': decision,
            'recommendation': recommendation
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        analysis = self.analyze_degradation()
        recommendation = self.make_recommendation(analysis)
        
        report = []
        report.append("="*80)
        report.append("PAPER TRADING POST-MORTEM ANALYSIS")
        report.append("="*80)
        
        report.append("\\nPERFORMANCE COMPARISON:")
        report.append(f"  Sharpe Ratio: {self.bt['sharpe']:.2f} → {self.pt['sharpe']:.2f} "
                     f"({analysis['sharpe_degradation_pct']:.0f}% degradation)")
        report.append(f"  Max Drawdown: {self.bt['max_dd']:.1%} → {self.pt['max_dd']:.1%}")
        report.append(f"  Win Rate: {self.bt['win_rate']:.1%} → {self.pt['win_rate']:.1%}")
        
        report.append("\\nROOT CAUSE ANALYSIS:")
        report.append(f"\\n1. Slippage Impact:")
        report.append(f"   {analysis['slippage']['explanation']}")
        
        report.append(f"\\n2. Fill Rate:")
        report.append(f"   {analysis['fill_rate']['impact']}")
        
        report.append(f"\\n3. Holding Period:")
        report.append(f"   {analysis['holding_period']['explanation']}")
        
        report.append(f"\\n4. Drawdown Control:")
        report.append(f"   {analysis['drawdown']['explanation']}")
        
        report.append("\\nRECOMMENDATION:")
        report.append(f"  Decision: {recommendation['decision']}")
        report.append(f"  Score: {recommendation['score']}/100")
        report.append(f"  {recommendation['recommendation']}")
        
        report.append("\\n" + "="*80)
        
        return "\\n".join(report)


# Example analysis
backtest = {
    'sharpe': 2.1,
    'max_dd': -0.12,
    'win_rate': 0.58,
    'avg_duration': 3.2
}

paper = {
    'sharpe': 0.9,
    'max_dd': -0.22,
    'win_rate': 0.52,
    'avg_duration': 4.1
}

analyzer = PaperTradingPostMortem(backtest, paper)
report = analyzer.generate_report()
print(report)
\`\`\`

**Decision Framework:**

Given the 57% Sharpe degradation and significantly higher slippage, the recommendation would be **CONDITIONAL** or **REJECT**.

**Specific Actions:**

1. **Don't proceed to live immediately**
2. **Modify strategy**:
   - Widen profit targets to compensate for slippage
   - Use market orders for smaller positions, limit orders for larger
   - Increase minimum expected return per trade
   - Reduce trading frequency to lower total costs
3. **Re-paper-trade** modified strategy for 2 months
4. **If re-paper-test succeeds**: Deploy 10% capital
5. **If it fails again**: Abandon strategy

**Adjusting Future Backtest Assumptions:**

\`\`\`python
# Update backtest parameters based on paper trading learnings

BACKTEST_PARAMS = {
    'slippage_bps': 15,  # Was 5, now 15 based on reality
    'commission_per_trade': 1.00,
    'commission_per_share': 0.005,
    'fill_rate': 0.92,  # Not all limit orders fill
    'market_impact_factor': 0.1,  # % of average volume
}
\`\`\`

---

## Question 2: Progressive Capital Allocation Strategy

**Scenario**: Your paper trading was successful. You're ready for live trading. Your firm has allocated $10M for the strategy. Your PM suggests: "Just deploy it all—we've validated it thoroughly."

**But you know better.** Design a progressive capital allocation strategy that:
1. Manages risk of unknown unknowns
2. Builds confidence gradually
3. Allows early detection of problems
4. Minimizes regret if strategy fails

### Comprehensive Answer

Progressive capital allocation is risk management 101. Never deploy full capital immediately.

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

@dataclass
class AllocationMilestone:
    """Capital allocation milestone"""
    stage: int
    capital_pct: float
    duration_days: int
    success_criteria: Dict[str, float]
    description: str

class ProgressiveAllocationStrategy:
    """
    Progressive capital allocation for new strategies
    """
    
    def __init__(
        self,
        total_capital: float,
        paper_trading_sharpe: float
    ):
        self.total_capital = total_capital
        self.paper_sharpe = paper_trading_sharpe
        
        # Define allocation milestones
        self.milestones = self._create_milestones()
        self.current_stage = 0
        self.start_date = None
    
    def _create_milestones(self) -> List[AllocationMilestone]:
        """
        Create progressive allocation schedule
        
        Stage 1: 10% capital, 30 days - Prove it works with real money
        Stage 2: 25% capital, 60 days - Build confidence
        Stage 3: 50% capital, 90 days - Scale up
        Stage 4: 75% capital, 90 days - Near full deployment
        Stage 5: 100% capital - Full deployment
        """
        
        milestones = [
            AllocationMilestone(
                stage=1,
                capital_pct=0.10,
                duration_days=30,
                success_criteria={
                    'min_sharpe': self.paper_sharpe * 0.7,  # Allow 30% degradation
                    'max_drawdown': 0.10,
                    'min_days_without_technical_issues': 30
                },
                description="Initial deployment - prove viability with real capital"
            ),
            AllocationMilestone(
                stage=2,
                capital_pct=0.25,
                duration_days=60,
                success_criteria={
                    'min_sharpe': self.paper_sharpe * 0.75,
                    'max_drawdown': 0.12,
                    'correlation_to_paper': 0.6  # Strategy behaving consistently
                },
                description="Scale to 25% - build confidence"
            ),
            AllocationMilestone(
                stage=3,
                capital_pct=0.50,
                duration_days=90,
                success_criteria={
                    'min_sharpe': self.paper_sharpe * 0.8,
                    'max_drawdown': 0.15,
                    'min_information_ratio': 0.5
                },
                description="Scale to 50% - meaningful capital deployed"
            ),
            AllocationMilestone(
                stage=4,
                capital_pct=0.75,
                duration_days=90,
                success_criteria={
                    'min_sharpe': self.paper_sharpe * 0.85,
                    'max_drawdown': 0.15,
                    'consistency_score': 0.7  # Consistent across time periods
                },
                description="Scale to 75% - approaching full deployment"
            ),
            AllocationMilestone(
                stage=5,
                capital_pct=1.00,
                duration_days=0,  # Ongoing
                success_criteria={
                    'min_sharpe': self.paper_sharpe * 0.8,
                    'max_drawdown': 0.20
                },
                description="Full deployment - continuous monitoring"
            )
        ]
        
        return milestones
    
    def start_live_trading(self) -> float:
        """Start live trading at Stage 1"""
        self.current_stage = 0
        self.start_date = datetime.now()
        
        initial_allocation = self.total_capital * self.milestones[0].capital_pct
        
        print(f"\\n{'='*80}")
        print(f"STARTING LIVE TRADING - STAGE 1")
        print(f"{'='*80}")
        print(f"Initial Allocation: \${initial_allocation:,.0f} ({self.milestones[0].capital_pct:.0%})")
        print(f"Total Capital: \${self.total_capital:,.0f}")
        print(f"Duration: {self.milestones[0].duration_days} days")
        print(f"Success Criteria:")
for criterion, value in self.milestones[0].success_criteria.items():
    print(f"  - {criterion}: {value}")
print(f"{'='*80}\\n")

return initial_allocation
    
    def evaluate_milestone(
    self,
    actual_performance: Dict[str, float]
) -> Dict:
"""
        Evaluate if milestone criteria met
        
        Returns decision: 'advance', 'hold', or 'reduce'
"""
current_milestone = self.milestones[self.current_stage]
criteria = current_milestone.success_criteria

results = {}
all_passed = True

for criterion, threshold in criteria.items():
    actual = actual_performance.get(criterion, 0)

if 'min_' in criterion:
    passed = actual >= threshold
            elif 'max_' in criterion:
passed = actual <= threshold
            else:
passed = actual >= threshold

results[criterion] = {
    'threshold': threshold,
    'actual': actual,
    'passed': passed
}

if not passed:
    all_passed = False
        
        # Make decision
if all_passed:
    decision = 'advance'
message = f"✓ All criteria met. Advance to Stage {self.current_stage + 2}"
        else:
            # Check severity of failures
critical_failures = [
    c for c, r in results.items()
                if not r['passed'] and('sharpe' in c or 'drawdown' in c)
            ]

if critical_failures:
    decision = 'reduce'
message = f"✗ Critical criteria failed. Reduce allocation by 50%"
            else:
decision = 'hold'
message = f"⚠ Some criteria not met. Hold current allocation, extend monitoring"

return {
    'decision': decision,
    'message': message,
    'results': results,
    'current_stage': self.current_stage + 1,
    'current_allocation_pct': current_milestone.capital_pct
}
    
    def advance_stage(self) -> Optional[float]:
"""Advance to next stage"""
if self.current_stage >= len(self.milestones) - 1:
    print("Already at full deployment")
return None

self.current_stage += 1
new_milestone = self.milestones[self.current_stage]
new_allocation = self.total_capital * new_milestone.capital_pct

print(f"\\n{'='*80}")
        print(f"ADVANCING TO STAGE {new_milestone.stage}")
        print(f"{'='*80}")
        print(f"New Allocation: \${new_allocation:,.0f} ({new_milestone.capital_pct:.0%})")
        print(f"Previous: \${self.total_capital * self.milestones[self.current_stage - 1].capital_pct:,.0f}")
        print(f"Increase: \${new_allocation - self.total_capital * self.milestones[self.current_stage - 1].capital_pct:,.0f}")
        print(f"Duration: {new_milestone.duration_days} days")
print(f"{'='*80}\\n")

return new_allocation


# Example usage
def example_progressive_allocation():
"""Example progressive allocation"""
    
    # Initialize with $10M total, paper Sharpe of 1.5
strategy = ProgressiveAllocationStrategy(
    total_capital = 10_000_000,
    paper_trading_sharpe = 1.5
)
    
    # Start live trading
initial_allocation = strategy.start_live_trading()
    
    # After 30 days, evaluate
actual_performance = {
    'min_sharpe': 1.2,  # 80 % of paper(1.5 * 0.8)
        'max_drawdown': 0.08,
    'min_days_without_technical_issues': 30
}

evaluation = strategy.evaluate_milestone(actual_performance)

print("\\nMILESTONE EVALUATION:")
print(f"Decision: {evaluation['decision']}")
print(f"Message: {evaluation['message']}")
print("\\nCriteria Results:")
for criterion, result in evaluation['results'].items():
    status = "✓" if result['passed'] else "✗"
print(f"  {status} {criterion}: {result['actual']} (threshold: {result['threshold']})")

if evaluation['decision'] == 'advance':
    new_allocation = strategy.advance_stage()
    print(f"\\nNew allocation deployed: \${new_allocation:,.0f}")

if __name__ == "__main__":
    example_progressive_allocation()
\`\`\`

**Key Principles:**

1. **Start Small (10%)**: Test with real money but limit downside
2. **Time Gates**: Minimum duration at each stage to observe various conditions
3. **Clear Criteria**: Objective metrics for advancement
4. **Reversibility**: Can scale back if performance degrades
5. **Patience**: Total deployment takes 270 days (~9 months)

**Why This Works:**
- Limits maximum regret if strategy fails
- Builds institutional confidence gradually
- Allows early problem detection with minimal capital at risk
- Provides multiple checkpoints for intervention

---

## Question 3: Emergency Shutdown Procedures

**At 2:47 AM, your monitoring system alerts: "Unusual trading activity detected. Strategy has executed 47 trades in last 10 minutes (normal: 2-3/day). P&L down 8% in 10 minutes."**

**What are your emergency procedures? Design a comprehensive incident response system.**

### Comprehensive Answer

\`\`\`python
class EmergencyShutdownSystem:
    """
    Emergency shutdown and incident response system
    """
    
    def __init__(self, trading_system: 'TradingSystem'):
        self.trading_system = trading_system
        self.circuit_breakers = self._initialize_circuit_breakers()
        self.incident_log = []
        self.shutdown_activated = False
    
    def _initialize_circuit_breakers(self) -> Dict:
        """Define circuit breaker rules"""
        return {
            'daily_loss_limit': {
                'threshold': -0.02,  # -2% daily loss
                'action': 'halt_trading',
                'severity': 'HIGH'
            },
            'drawdown_limit': {
                'threshold': -0.05,  # -5% from peak
                'action': 'halt_trading',
                'severity': 'CRITICAL'
            },
            'trade_velocity': {
                'threshold': 10,  # 10x normal trade rate
                'window_minutes': 10,
                'action': 'halt_trading',
                'severity': 'HIGH'
            },
            'position_size_violation': {
                'threshold': 1.5,  # 1.5x max position size
                'action': 'close_position',
                'severity': 'MEDIUM'
            },
            'data_feed_stale': {
                'threshold_seconds': 30,
                'action': 'halt_trading',
                'severity': 'CRITICAL'
            }
        }
    
    async def emergency_shutdown(self, reason: str, severity: str):
        """
        Execute emergency shutdown
        
        Steps:
        1. Halt all new trading immediately
        2. Cancel pending orders
        3. Close positions (depending on severity)
        4. Alert team
        5. Log incident
        6. Enter safe mode
        """
        print(f"\\n🚨 EMERGENCY SHUTDOWN INITIATED 🚨")
        print(f"Reason: {reason}")
        print(f"Severity: {severity}")
        print(f"Time: {datetime.now()}")
        
        self.shutdown_activated = True
        
        # Step 1: Halt new trading
        self.trading_system.halt_trading()
        print("✓ New trading halted")
        
        # Step 2: Cancel all pending orders
        cancelled = await self.trading_system.cancel_all_orders()
        print(f"✓ Cancelled {cancelled} pending orders")
        
        # Step 3: Close positions if critical
        if severity == 'CRITICAL':
            closed = await self.trading_system.close_all_positions()
            print(f"✓ Closed {len(closed)} positions")
        
        # Step 4: Alert team
        await self._send_alerts(reason, severity)
        print("✓ Team alerted")
        
        # Step 5: Log incident
        self._log_incident(reason, severity)
        print("✓ Incident logged")
        
        # Step 6: Enter safe mode
        self.trading_system.enter_safe_mode()
        print("✓ System in safe mode")
        
        print(f"\\nEmergency shutdown complete. System halted.\\n")
    
    async def _send_alerts(self, reason: str, severity: str):
        """Send alerts via multiple channels"""
        # Email
        # SMS
        # PagerDuty
        # Slack
        pass
    
    def _log_incident(self, reason: str, severity: str):
        """Log incident for post-mortem"""
        incident = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'severity': severity,
            'portfolio_state': self.trading_system.get_portfolio_snapshot(),
            'recent_trades': self.trading_system.get_recent_trades(n=50)
        }
        self.incident_log.append(incident)
\`\`\`

**Incident Response Checklist:**
1. [ ] Emergency shutdown triggered
2. [ ] Team notified (phone, email, Slack)
3. [ ] Positions assessed
4. [ ] Root cause investigation started
5. [ ] Incident report filed
6. [ ] Post-mortem scheduled
7. [ ] Strategy paused pending review
8. [ ] Restart procedure defined

**Prevention > Response**: Better to have false alarms than miss a real emergency.
`,
        },
    ],
};

export default paperTradingVsLiveDiscussion;
