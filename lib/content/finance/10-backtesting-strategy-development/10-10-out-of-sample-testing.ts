import { Content } from '@/lib/types';

const outOfSampleTesting: Content = {
  title: 'Out-of-Sample Testing',
  description:
    'Master proper train/test/validation splits, hold-out period requirements, paper trading transitions, and performance degradation analysis for robust strategy validation',
  sections: [
    {
      title: 'The Gold Standard of Strategy Validation',
      content: `
# The Gold Standard of Strategy Validation

Out-of-sample (OOS) testing is the most critical validation step—it's where strategies prove they can work on data they've never seen before.

## Why Out-of-Sample Testing Matters

**Case Study - LTCM Redux**: A major quantitative fund developed a high-frequency strategy showing a 3.2 Sharpe ratio over 3 years of historical data. They deployed $50M. Within 2 months, they'd lost $12M. 

**Post-mortem**: The entire 3-year period was used for optimization. There was no true out-of-sample test. The strategy had memorized historical patterns that didn't repeat.

### The Three-Way Split

Professional quant firms use a rigorous three-way data split:

1. **Training Set (50-60%)**: Used for strategy development and parameter optimization
2. **Validation Set (20-25%)**: Used for model selection and hyperparameter tuning
3. **Test Set (20-25%)**: NEVER TOUCHED until final validation

**Critical Rule**: The test set must remain completely untouched during all development. Once you look at it, it's compromised.

## Implementation

\`\`\`python
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import hashlib
import json

@dataclass
class DataSplit:
    """Represents a data split"""
    name: str
    start_date: datetime
    end_date: datetime
    data: pd.DataFrame
    is_locked: bool = False
    access_count: int = 0
    
class OutOfSampleTester:
    """
    Rigorous out-of-sample testing framework
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        train_pct: float = 0.60,
        val_pct: float = 0.20,
        test_pct: float = 0.20
    ):
        """
        Initialize with complete dataset
        
        Args:
            data: Complete historical data
            train_pct: Training set percentage
            val_pct: Validation set percentage
            test_pct: Test set percentage (held out)
        """
        if not np.isclose(train_pct + val_pct + test_pct, 1.0):
            raise ValueError("Percentages must sum to 1.0")
        
        self.data = data.copy()
        self.splits: Dict[str, DataSplit] = {}
        
        # Create splits
        self._create_splits(train_pct, val_pct, test_pct)
        
        # Lock test set immediately
        self.splits['test'].is_locked = True
        
    def _create_splits(
        self,
        train_pct: float,
        val_pct: float,
        test_pct: float
    ):
        """Create train/validation/test splits"""
        
        n = len(self.data)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        # Training set
        self.splits['train'] = DataSplit(
            name='train',
            start_date=self.data.index[0],
            end_date=self.data.index[train_end - 1],
            data=self.data.iloc[:train_end].copy(),
            is_locked=False
        )
        
        # Validation set
        self.splits['validation'] = DataSplit(
            name='validation',
            start_date=self.data.index[train_end],
            end_date=self.data.index[val_end - 1],
            data=self.data.iloc[train_end:val_end].copy(),
            is_locked=False
        )
        
        # Test set (LOCKED)
        self.splits['test'] = DataSplit(
            name='test',
            start_date=self.data.index[val_end],
            end_date=self.data.index[-1],
            data=self.data.iloc[val_end:].copy(),
            is_locked=True  # Immediately locked
        )
        
        # Create integrity hashes
        self._create_integrity_hashes()
        
        print("\\n" + "="*80)
        print("DATA SPLITS CREATED")
        print("="*80)
        for split_name, split in self.splits.items():
            print(f"\\n{split_name.upper()}:")
            print(f"  Period: {split.start_date.date()} to {split.end_date.date()}")
            print(f"  Observations: {len(split.data)}")
            print(f"  Locked: {'🔒 YES' if split.is_locked else '✓ No'}")
        
        print("\\n⚠️  TEST SET IS LOCKED - DO NOT ACCESS UNTIL FINAL VALIDATION")
        print("="*80 + "\\n")
    
    def _create_integrity_hashes(self):
        """Create cryptographic hashes for data integrity"""
        for split_name, split in self.splits.items():
            data_json = split.data.to_json()
            hash_value = hashlib.sha256(data_json.encode()).hexdigest()
            setattr(split, 'hash', hash_value)
    
    def get_train_data(self) -> pd.DataFrame:
        """Get training data"""
        return self.splits['train'].data.copy()
    
    def get_validation_data(self) -> pd.DataFrame:
        """Get validation data"""
        return self.splits['validation'].data.copy()
    
    def request_test_data(
        self,
        requester: str,
        justification: str,
        strategy_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Request access to locked test set
        
        THIS SHOULD ONLY BE CALLED ONCE FOR FINAL VALIDATION
        
        Args:
            requester: Name of person requesting access
            justification: Justification for access
            strategy_name: Strategy being tested
            
        Returns:
            Test data if access granted, None otherwise
        """
        test_split = self.splits['test']
        
        if test_split.access_count > 0:
            print("\\n" + "="*80)
            print("❌ TEST SET ACCESS DENIED")
            print("="*80)
            print(f"Test set has already been accessed {test_split.access_count} time(s)")
            print("Data is compromised. Cannot grant additional access.")
            print("="*80 + "\\n")
            return None
        
        # Log access
        test_split.access_count += 1
        
        print("\\n" + "="*80)
        print("🔓 TEST SET ACCESS GRANTED")
        print("="*80)
        print(f"Requester: {requester}")
        print(f"Strategy: {strategy_name}")
        print(f"Justification: {justification}")
        print(f"Access granted at: {datetime.now()}")
        print(f"\\n⚠️  THIS IS THE ONLY ALLOWED ACCESS TO TEST SET")
        print(f"⚠️  Results must be reported regardless of outcome")
        print("="*80 + "\\n")
        
        return test_split.data.copy()
    
    def validate_final_performance(
        self,
        strategy_func: callable,
        params: Dict,
        requester: str,
        strategy_name: str
    ) -> Dict:
        """
        Final validation on test set
        
        This is the ONLY function that should access test data
        
        Args:
            strategy_func: Strategy function
            params: Final parameters from validation
            requester: Person conducting validation
            strategy_name: Strategy name
            
        Returns:
            Validation results
        """
        # Get test data (will be logged)
        test_data = self.request_test_data(
            requester=requester,
            justification="Final strategy validation",
            strategy_name=strategy_name
        )
        
        if test_data is None:
            return {'error': 'Test set access denied'}
        
        # Run strategy on test data
        test_returns = strategy_func(test_data, params)
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(test_returns)
        
        # Also get in-sample and validation metrics for comparison
        train_returns = strategy_func(self.get_train_data(), params)
        val_returns = strategy_func(self.get_validation_data(), params)
        
        train_metrics = self._calculate_metrics(train_returns)
        val_metrics = self._calculate_metrics(val_returns)
        
        # Calculate degradation
        degradation = self._calculate_degradation(
            train_metrics,
            val_metrics,
            test_metrics
        )
        
        # Make final decision
        decision = self._make_final_decision(
            train_metrics,
            val_metrics,
            test_metrics,
            degradation
        )
        
        report = {
            'strategy_name': strategy_name,
            'requester': requester,
            'validation_date': datetime.now(),
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'degradation': degradation,
            'decision': decision,
            'test_set_access_count': self.splits['test'].access_count
        }
        
        self._print_final_report(report)
        
        return report
    
    def _calculate_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        if len(returns) == 0:
            return {'sharpe': 0, 'total_return': 0, 'max_dd': 0}
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        total_return = np.sum(returns)
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': (returns > 0).mean()
        }
    
    def _calculate_degradation(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict
    ) -> Dict:
        """Calculate performance degradation"""
        
        def pct_change(from_val, to_val):
            if from_val == 0:
                return 0
            return (from_val - to_val) / abs(from_val) * 100
        
        return {
            'train_to_val_sharpe': pct_change(
                train_metrics['sharpe'],
                val_metrics['sharpe']
            ),
            'train_to_test_sharpe': pct_change(
                train_metrics['sharpe'],
                test_metrics['sharpe']
            ),
            'val_to_test_sharpe': pct_change(
                val_metrics['sharpe'],
                test_metrics['sharpe']
            )
        }
    
    def _make_final_decision(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict,
        degradation: Dict
    ) -> Dict:
        """Make final deployment decision"""
        
        # Criteria for approval
        test_sharpe = test_metrics['sharpe']
        total_degradation = degradation['train_to_test_sharpe']
        
        if test_sharpe >= 1.0 and total_degradation < 30:
            status = "APPROVED"
            reason = f"Test Sharpe {test_sharpe:.2f} exceeds threshold, degradation {total_degradation:.1f}% acceptable"
            recommendation = "Proceed to paper trading"
        elif test_sharpe >= 0.5 and total_degradation < 50:
            status = "CONDITIONAL"
            reason = f"Test Sharpe {test_sharpe:.2f} marginal, degradation {total_degradation:.1f}%"
            recommendation = "Extended paper trading required, reduce allocation"
        else:
            status = "REJECTED"
            reason = f"Test Sharpe {test_sharpe:.2f} insufficient or degradation {total_degradation:.1f}% excessive"
            recommendation = "Do not deploy"
        
        return {
            'status': status,
            'reason': reason,
            'recommendation': recommendation
        }
    
    def _print_final_report(self, report: Dict):
        """Print final validation report"""
        print("\\n" + "="*80)
        print("FINAL OUT-OF-SAMPLE VALIDATION REPORT")
        print("="*80)
        print(f"Strategy: {report['strategy_name']}")
        print(f"Validated by: {report['requester']}")
        print(f"Date: {report['validation_date']}")
        print("")
        
        print("Performance Summary:")
        print(f"{'Metric':<20} {'Training':<15} {'Validation':<15} {'Test':<15}")
        print("-" * 70)
        
        metrics = ['sharpe', 'total_return', 'max_dd', 'win_rate']
        for metric in metrics:
            train_val = report['train_metrics'][metric]
            val_val = report['validation_metrics'][metric]
            test_val = report['test_metrics'][metric]
            
            print(f"{metric:<20} {train_val:<15.3f} {val_val:<15.3f} {test_val:<15.3f}")
        
        print("")
        print("Degradation Analysis:")
        for key, value in report['degradation'].items():
            print(f"  {key}: {value:.1f}%")
        
        print("")
        print("FINAL DECISION:")
        print(f"  Status: {report['decision']['status']}")
        print(f"  Reason: {report['decision']['reason']}")
        print(f"  Recommendation: {report['decision']['recommendation']}")
        print("")
        print("="*80 + "\\n")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
    
    data = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    # Initialize OOS tester
    oos_tester = OutOfSampleTester(
        data=data,
        train_pct=0.60,
        val_pct=0.20,
        test_pct=0.20
    )
    
    # Example strategy
    def simple_strategy(data: pd.DataFrame, params: Dict) -> np.ndarray:
        """Simple moving average strategy"""
        period = int(params['period'])
        ma = data['close'].rolling(period).mean()
        position = np.where(data['close'] > ma, 1, -1)
        returns = data['close'].pct_change() * np.roll(position, 1)
        return returns[period:].values
    
    # Final validation
    final_params = {'period': 30}
    
    report = oos_tester.validate_final_performance(
        strategy_func=simple_strategy,
        params=final_params,
        requester="John Doe",
        strategy_name="SMA Crossover v1.0"
    )
\`\`\`

## Key Principles

### 1. **One-Time Test Set Access**
- Test set can only be accessed ONCE
- After access, data is considered compromised
- Results must be reported regardless of outcome

### 2. **Minimum Hold-Out Periods**
- At least 20% of data for testing
- Minimum 6 months for daily strategies
- Minimum 2 years for lower-frequency strategies

### 3. **Performance Degradation Limits**
- <10% degradation: Excellent
- 10-30%: Acceptable
- 30-50%: Concerning
- >50%: Reject

### 4. **Statistical Significance on Test Set**
- Must be significant at p < 0.05
- Account for multiple testing if screening many strategies
- Use confidence intervals, not just point estimates

## Common Mistakes

1. ❌ **Peeking at test set** during development
2. ❌ **Multiple test set accesses** (each look contaminates data)
3. ❌ **Cherry-picking test periods** (must be predetermined)
4. ❌ **Inadequate hold-out period** (<20% or <6 months)
5. ❌ **Not reporting failed OOS tests** (publication bias)

## Production Checklist

- [ ] Test set locked before any development
- [ ] Test set >= 20% of data and >= 6 months
- [ ] Cryptographic hash verification of test data
- [ ] Logged access control for test set
- [ ] Final validation documented with requester and timestamp
- [ ] Results reported regardless of outcome
- [ ] Decision criteria specified before OOS test
- [ ] Investment committee approval before live trading
`,
    },
  ],
};

export default outOfSampleTesting;
