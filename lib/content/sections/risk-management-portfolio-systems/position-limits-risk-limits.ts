export const positionLimitsRiskLimits = `
# Position Limits and Risk Limits

## Introduction

"Risk limits are like seatbelts - inconvenient until you need them."

The JPMorgan "London Whale" (2012) lost $6.2B because risk limits were ignored. Knight Capital (2012) went bankrupt in 45 minutes when trading limits failed. Effective risk limits are the last line of defense against catastrophic losses.

This section covers designing, implementing, and enforcing comprehensive limit frameworks that prevent disasters while allowing profitable trading.

## Types of Risk Limits

### 1. Position Limits

Maximum exposure to any single instrument:

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class LimitType(Enum):
    SOFT = "SOFT"  # Warning, can override
    HARD = "HARD"  # Cannot be exceeded
    
@dataclass
class RiskLimit:
    """Single risk limit definition"""
    limit_id: str
    description: str
    limit_type: LimitType
    threshold: float
    current_value: float = 0.0
    
    @property
    def utilization(self) -> float:
        """Utilization as percentage"""
        return (self.current_value / self.threshold * 100) if self.threshold > 0 else 0
    
    @property
    def available(self) -> float:
        """Remaining capacity"""
        return max(0, self.threshold - self.current_value)
    
    @property
    def breached(self) -> bool:
        """Is limit breached?"""
        return self.current_value > self.threshold

class RiskLimitFramework:
    """
    Comprehensive risk limit management system
    """
    def __init__(self):
        self.limits = {}
        self.breaches = []
        
    def add_limit(self, limit: RiskLimit):
        """Add new limit"""
        self.limits[limit.limit_id] = limit
        
    def check_pre_trade(self, 
                       trade: Dict,
                       position_limits: Dict[str, float],
                       risk_limits: Dict[str, float]) -> Dict:
        """
        Pre-trade risk check
        
        Returns: {approved: bool, violations: list, warnings: list}
        """
        violations = []
        warnings = []
        
        # Position size check
        symbol = trade.get('symbol')
        quantity = trade.get('quantity', 0)
        
        if symbol in position_limits:
            current_position = trade.get('current_position', 0)
            new_position = current_position + quantity
            limit = position_limits[symbol]
            
            if abs(new_position) > limit:
                violations.append({
                    'type': 'POSITION_LIMIT',
                    'symbol': symbol,
                    'new_position': new_position,
                    'limit': limit,
                    'excess': abs(new_position) - limit
                })
        
        # Trade size check
        trade_value = abs(quantity * trade.get('price', 0))
        max_trade_size = risk_limits.get('max_trade_size', float('inf'))
        
        if trade_value > max_trade_size:
            violations.append({
                'type': 'TRADE_SIZE',
                'trade_value': trade_value,
                'limit': max_trade_size
            })
        
        # Concentration check
        portfolio_value = trade.get('portfolio_value', 1)
        position_value = abs(new_position * trade.get('price', 0))
        concentration = position_value / portfolio_value
        max_concentration = risk_limits.get('max_concentration', 1.0)
        
        if concentration > max_concentration:
            warnings.append({
                'type': 'CONCENTRATION',
                'concentration_pct': concentration * 100,
                'limit_pct': max_concentration * 100
            })
        
        approved = len(violations) == 0
        
        return {
            'approved': approved,
            'violations': violations,
            'warnings': warnings,
            'trade_id': trade.get('trade_id', 'unknown')
        }
    
    def check_all_limits(self) -> Dict:
        """
        Check all limits for breaches
        """
        breached_limits = []
        warning_limits = []
        
        for limit_id, limit in self.limits.items():
            if limit.breached:
                if limit.limit_type == LimitType.HARD:
                    breached_limits.append({
                        'limit_id': limit_id,
                        'description': limit.description,
                        'current': limit.current_value,
                        'threshold': limit.threshold,
                        'excess': limit.current_value - limit.threshold,
                        'type': 'HARD_BREACH'
                    })
                else:
                    warning_limits.append({
                        'limit_id': limit_id,
                        'description': limit.description,
                        'current': limit.current_value,
                        'threshold': limit.threshold,
                        'excess': limit.current_value - limit.threshold,
                        'type': 'SOFT_BREACH'
                    })
            elif limit.utilization > 80:  # Warning at 80% utilization
                warning_limits.append({
                    'limit_id': limit_id,
                    'description': limit.description,
                    'utilization': limit.utilization,
                    'type': 'HIGH_UTILIZATION'
                })
        
        return {
            'hard_breaches': breached_limits,
            'warnings': warning_limits,
            'total_limits': len(self.limits),
            'status': 'BREACH' if breached_limits else ('WARNING' if warning_limits else 'OK')
        }
    
    def update_limit_value(self, limit_id: str, new_value: float):
        """Update current value for limit"""
        if limit_id in self.limits:
            old_value = self.limits[limit_id].current_value
            self.limits[limit_id].current_value = new_value
            
            # Check if breach occurred
            if not self.limits[limit_id].breached and new_value > self.limits[limit_id].threshold:
                self.record_breach(limit_id, old_value, new_value)
    
    def record_breach(self, limit_id: str, old_value: float, new_value: float):
        """Record limit breach"""
        limit = self.limits[limit_id]
        
        breach = {
            'timestamp': datetime.now(),
            'limit_id': limit_id,
            'description': limit.description,
            'limit_type': limit.limit_type.value,
            'threshold': limit.threshold,
            'old_value': old_value,
            'new_value': new_value,
            'excess': new_value - limit.threshold
        }
        
        self.breaches.append(breach)
        
        # Alert
        self.send_alert(breach)
    
    def send_alert(self, breach: Dict):
        """Send alert for limit breach"""
        print(f"🚨 LIMIT BREACH ALERT")
        print(f"   Limit: {breach['description']}")
        print(f"   Type: {breach['limit_type']}")
        print(f"   Threshold: {breach['threshold']:,.0f}")
        print(f"   Current: {breach['new_value']:,.0f}")
        print(f"   Excess: {breach['excess']:,.0f}")
        print(f"   Time: {breach['timestamp']}")

# Example Usage
if __name__ == "__main__":
    # Create limit framework
    framework = RiskLimitFramework()
    
    # Add limits
    framework.add_limit(RiskLimit(
        limit_id='POSITION_AAPL',
        description='Maximum AAPL position',
        limit_type=LimitType.HARD,
        threshold=100000,  # 100k shares
        current_value=85000
    ))
    
    framework.add_limit(RiskLimit(
        limit_id='DAILY_VAR',
        description='Daily 99% VaR',
        limit_type=LimitType.HARD,
        threshold=5000000,  # $5M
        current_value=4200000
    ))
    
    framework.add_limit(RiskLimit(
        limit_id='PORTFOLIO_DELTA',
        description='Net portfolio delta',
        limit_type=LimitType.SOFT,
        threshold=1000000,
        current_value=850000
    ))
    
    print("Risk Limit Framework")
    print("="*70)
    print()
    
    # Check all limits
    status = framework.check_all_limits()
    print(f"Overall Status: {status['status']}")
    print(f"Total Limits: {status['total_limits']}")
    
    if status['warnings']:
        print()
        print("Warnings:")
        for warning in status['warnings']:
            print(f"  - {warning['description']}: {warning.get('utilization', 0):.1f}% utilized")
    
    print()
    
    # Pre-trade check
    trade = {
        'trade_id': 'TRADE001',
        'symbol': 'AAPL',
        'quantity': 20000,
        'price': 180.0,
        'current_position': 85000,
        'portfolio_value': 50000000
    }
    
    position_limits = {'AAPL': 100000}
    risk_limits = {
        'max_trade_size': 5000000,
        'max_concentration': 0.10
    }
    
    result = framework.check_pre_trade(trade, position_limits, risk_limits)
    
    print(f"Pre-Trade Check for {trade['trade_id']}:")
    print(f"  Approved: {result['approved']}")
    
    if result['violations']:
        print("  Violations:")
        for v in result['violations']:
            print(f"    - {v['type']}: {v}")
    
    if result['warnings']:
        print("  Warnings:")
        for w in result['warnings']:
            print(f"    - {w['type']}: {w}")
\`\`\`

## Loss Limits and Stop-Loss Rules

\`\`\`python
class LossLimitManager:
    """
    Manage loss limits and automatic stop-loss
    """
    def __init__(self, 
                 daily_loss_limit: float,
                 monthly_loss_limit: float,
                 drawdown_limit: float):
        """
        Args:
            daily_loss_limit: Maximum daily loss
            monthly_loss_limit: Maximum monthly loss  
            drawdown_limit: Maximum drawdown from peak
        """
        self.daily_limit = daily_loss_limit
        self.monthly_limit = monthly_loss_limit
        self.drawdown_limit = drawdown_limit
        
        self.daily_pnl = 0
        self.monthly_pnl = 0
        self.peak_equity = 0
        self.current_equity = 0
        
        self.stop_trading = False
        
    def update_pnl(self, pnl_change: float):
        """Update P&L and check limits"""
        self.daily_pnl += pnl_change
        self.monthly_pnl += pnl_change
        self.current_equity += pnl_change
        
        # Update peak
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        
        # Check limits
        self.check_loss_limits()
    
    def check_loss_limits(self) -> Dict:
        """Check if any loss limits breached"""
        breaches = []
        
        # Daily loss limit
        if self.daily_pnl < -self.daily_limit:
            breaches.append({
                'type': 'DAILY_LOSS',
                'pnl': self.daily_pnl,
                'limit': -self.daily_limit,
                'action': 'STOP_TRADING'
            })
            self.stop_trading = True
        
        # Monthly loss limit
        if self.monthly_pnl < -self.monthly_limit:
            breaches.append({
                'type': 'MONTHLY_LOSS',
                'pnl': self.monthly_pnl,
                'limit': -self.monthly_limit,
                'action': 'STOP_TRADING'
            })
            self.stop_trading = True
        
        # Drawdown limit
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > self.drawdown_limit:
            breaches.append({
                'type': 'DRAWDOWN',
                'drawdown_pct': drawdown * 100,
                'limit_pct': self.drawdown_limit * 100,
                'action': 'STOP_TRADING'
            })
            self.stop_trading = True
        
        if breaches:
            for breach in breaches:
                print(f"🛑 LOSS LIMIT BREACH: {breach['type']}")
                print(f"   Action: {breach['action']}")
        
        return {
            'breaches': breaches,
            'stop_trading': self.stop_trading,
            'daily_pnl': self.daily_pnl,
            'monthly_pnl': self.monthly_pnl,
            'drawdown_pct': drawdown * 100 if self.peak_equity > 0 else 0
        }
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return not self.stop_trading
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0
        # Don't reset stop_trading automatically - requires manual override
    
    def reset_monthly(self):
        """Reset monthly counters"""
        self.monthly_pnl = 0

# Example
if __name__ == "__main__":
    loss_mgr = LossLimitManager(
        daily_loss_limit=500000,    # $500K daily
        monthly_loss_limit=2000000,  # $2M monthly
        drawdown_limit=0.15          # 15% max drawdown
    )
    
    # Set initial equity
    loss_mgr.current_equity = 10000000  # $10M
    loss_mgr.peak_equity = 10000000
    
    print("Loss Limit Management")
    print("="*70)
    print()
    
    # Simulate losing day
    print("Simulating bad trading day...")
    loss_mgr.update_pnl(-200000)  # -$200K
    loss_mgr.update_pnl(-150000)  # -$150K
    loss_mgr.update_pnl(-180000)  # -$180K
    
    status = loss_mgr.check_loss_limits()
    
    print(f"Daily P&L: \${status['daily_pnl']:,.0f}")
print(f"Can Trade: {loss_mgr.can_trade()}")
\`\`\`

## Concentration Limits

\`\`\`python
class ConcentrationLimitMonitor:
    """
    Monitor concentration risk limits
    """
    def __init__(self,
                 max_single_position: float = 0.05,
                 max_sector: float = 0.25,
                 max_country: float = 0.30):
        """
        Args:
            max_single_position: Max % in single security
            max_sector: Max % in single sector
            max_country: Max % in single country
        """
        self.max_single = max_single_position
        self.max_sector = max_sector
        self.max_country = max_country
        
    def check_concentrations(self,
                            positions: Dict[str, Dict]) -> Dict:
        """
        Check all concentration limits
        
        Args:
            positions: {symbol: {value, sector, country}}
        """
        # Calculate total portfolio value
        total_value = sum(p['value'] for p in positions.values())
        
        violations = []
        
        # Single position concentration
        for symbol, info in positions.items():
            concentration = info['value'] / total_value
            if concentration > self.max_single:
                violations.append({
                    'type': 'SINGLE_POSITION',
                    'symbol': symbol,
                    'concentration_pct': concentration * 100,
                    'limit_pct': self.max_single * 100,
                    'excess_pct': (concentration - self.max_single) * 100
                })
        
        # Sector concentration
        sector_exposure = {}
        for symbol, info in positions.items():
            sector = info.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + info['value']
        
        for sector, value in sector_exposure.items():
            concentration = value / total_value
            if concentration > self.max_sector:
                violations.append({
                    'type': 'SECTOR',
                    'sector': sector,
                    'concentration_pct': concentration * 100,
                    'limit_pct': self.max_sector * 100,
                    'excess_pct': (concentration - self.max_sector) * 100
                })
        
        # Country concentration
        country_exposure = {}
        for symbol, info in positions.items():
            country = info.get('country', 'Unknown')
            country_exposure[country] = country_exposure.get(country, 0) + info['value']
        
        for country, value in country_exposure.items():
            concentration = value / total_value
            if concentration > self.max_country:
                violations.append({
                    'type': 'COUNTRY',
                    'country': country,
                    'concentration_pct': concentration * 100,
                    'limit_pct': self.max_country * 100,
                    'excess_pct': (concentration - self.max_country) * 100
                })
        
        return {
            'violations': violations,
            'compliant': len(violations) == 0,
            'total_portfolio_value': total_value
        }
\`\`\`

## Real-World: JPMorgan London Whale

**What Happened** (2012):
- Chief Investment Office (CIO) took large synthetic credit positions
- VaR model changed to reduce reported risk
- Position grew to $157B notional
- Lost $6.2B when unable to unwind

**Limit Failures**:
1. VaR limits circumvented by model change
2. Concentration limits ignored
3. Loss limits breached without stopping
4. Management override of risk controls

**Lessons**:
- Independent risk validation critical
- Cannot change models to fit desired outcome
- Hard limits must be hard
- Regular limit framework review

## Key Takeaways

1. **Pre-Trade Checks**: Prevent limit breaches before they happen
2. **Hard vs Soft**: Hard limits cannot be exceeded
3. **Multiple Layers**: Position, risk factor, loss, concentration
4. **Real-Time**: Limits must be checked continuously
5. **Escalation**: Clear process for limit breaches
6. **Independent**: Risk function must be independent
7. **Regular Review**: Limits should evolve with strategy

## Conclusion

Effective risk limits are the safety net that prevents catastrophic losses. They must be:
- **Comprehensive**: Cover all risk types
- **Enforced**: Technology prevents violations
- **Independent**: Cannot be overridden by traders
- **Dynamic**: Adjust to changing conditions
- **Monitored**: Real-time tracking and alerts

As the London Whale showed, circumventing limits leads to disaster. Strong limit frameworks save firms from themselves.

Next: Real-Time Risk Monitoring - continuous risk surveillance systems.
`;

