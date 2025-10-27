import { Content } from '@/lib/types';

const timeSeriesCrossValidation: Content = {
  title: 'Cross-Validation for Time Series',
  description:
    'Master time series-specific cross-validation techniques including purging, embargo, combinatorial CV, and walk-forward validation to prevent temporal leakage',
  sections: [
    {
      title: 'The Time Series Cross-Validation Challenge',
      content: `
# Cross-Validation for Time Series

Traditional k-fold cross-validation doesn't work for time series—it violates temporal ordering and creates look-ahead bias. Time series requires specialized techniques.

## The Problem with Standard Cross-Validation

\`\`\`python
# WRONG for time series
from sklearn.model_selection import KFold

# This shuffles data randomly, violating time order!
kfold = KFold(n_splits=5, shuffle=True)  # ❌ Look-ahead bias
\`\`\`

**The Issue**: In standard K-fold CV, future data can appear in training sets when predicting past data, creating impossible information leakage.

## Time Series-Specific Techniques

### 1. Walk-Forward Cross-Validation

\`\`\`python
from typing import List, Tuple, Iterator
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CVFold:
    """Represents a single CV fold"""
    fold_num: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_dates: List[datetime]
    test_dates: List[datetime]

class TimeSeriesCV:
    """
    Time series cross-validation with purging and embargo
    
    Based on "Advances in Financial Machine Learning" by Marcos López de Prado
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,  # 1 year
        purge_gap: int = 5,  # Days to purge after train
        embargo_gap: int = 2,  # Days to embargo before test
        min_train_size: int = 504  # Minimum 2 years training
    ):
        """
        Initialize time series cross-validator
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set in each split
            purge_gap: Days to remove after training set
            embargo_gap: Days to remove before test set
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        self.min_train_size = min_train_size
    
    def split(
        self,
        data: pd.DataFrame,
        expanding: bool = True
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging and embargo
        
        Args:
            data: DataFrame with DatetimeIndex
            expanding: If True, use expanding window; if False, use rolling window
            
        Yields:
            (train_indices, test_indices) tuples
        """
        n = len(data)
        
        # Calculate split points
        total_test_size = self.test_size + self.purge_gap + self.embargo_gap
        step_size = (n - self.min_train_size - total_test_size) // (self.n_splits - 1)
        
        for i in range(self.n_splits):
            # Test set placement
            test_end = n - i * step_size
            test_start = test_end - self.test_size
            
            # Apply embargo before test
            embargo_start = test_start - self.embargo_gap
            
            # Training set
            if expanding:
                # Expanding window: train from start
                train_start = 0
                train_end = embargo_start - self.purge_gap
            else:
                # Rolling window: fixed-size training window
                train_end = embargo_start - self.purge_gap
                train_start = max(0, train_end - self.min_train_size)
            
            # Ensure minimum training size
            if train_end - train_start < self.min_train_size:
                continue
            
            # Ensure test set is valid
            if test_start < 0 or test_end > n:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_fold_info(
        self,
        data: pd.DataFrame,
        expanding: bool = True
    ) -> List[CVFold]:
        """
        Get detailed information about each fold
        
        Args:
            data: DataFrame with DatetimeIndex
            expanding: If True, use expanding window
            
        Returns:
            List of CVFold objects
        """
        folds = []
        
        for fold_num, (train_idx, test_idx) in enumerate(self.split(data, expanding), 1):
            fold = CVFold(
                fold_num=fold_num,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                train_dates=data.index[train_idx].tolist(),
                test_dates=data.index[test_idx].tolist()
            )
            folds.append(fold)
        
        return folds
    
    def visualize_splits(
        self,
        data: pd.DataFrame,
        expanding: bool = True
    ):
        """Visualize CV splits"""
        import matplotlib.pyplot as plt
        
        folds = self.get_fold_info(data, expanding)
        
        fig, ax = plt.subplots(figsize=(15, len(folds) * 0.8))
        
        for i, fold in enumerate(folds):
            # Plot training period
            ax.barh(
                i, 
                fold.train_end - fold.train_start,
                left=fold.train_start,
                height=0.5,
                color='blue',
                alpha=0.6,
                label='Train' if i == 0 else ''
            )
            
            # Plot purge period
            purge_start = fold.train_end
            purge_end = fold.train_end + self.purge_gap
            ax.barh(
                i,
                purge_end - purge_start,
                left=purge_start,
                height=0.5,
                color='red',
                alpha=0.3,
                label='Purge' if i == 0 else ''
            )
            
            # Plot embargo period
            embargo_start = fold.test_start - self.embargo_gap
            embargo_end = fold.test_start
            ax.barh(
                i,
                embargo_end - embargo_start,
                left=embargo_start,
                height=0.5,
                color='orange',
                alpha=0.3,
                label='Embargo' if i == 0 else ''
            )
            
            # Plot test period
            ax.barh(
                i,
                fold.test_end - fold.test_start,
                left=fold.test_start,
                height=0.5,
                color='green',
                alpha=0.6,
                label='Test' if i == 0 else ''
            )
        
        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels([f'Fold {f.fold_num}' for f in folds])
        ax.set_xlabel('Time Index')
        ax.set_title(f'Time Series Cross-Validation Splits\\n({"Expanding" if expanding else "Rolling"} Window)')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('cv_splits.png', dpi=300, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range('2015-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'returns': np.random.randn(len(dates)) * 0.01
    }, index=dates)
    
    # Initialize CV
    cv = TimeSeriesCV(
        n_splits=5,
        test_size=252,  # 1 year test
        purge_gap=5,  # 1 week purge
        embargo_gap=2  # 2 days embargo
    )
    
    # Get fold information
    print("\\nTIME SERIES CROSS-VALIDATION FOLDS")
    print("="*80)
    
    folds = cv.get_fold_info(data, expanding=True)
    
    for fold in folds:
        print(f"\\nFold {fold.fold_num}:")
        print(f"  Train: {fold.train_dates[0].date()} to {fold.train_dates[-1].date()} ({len(fold.train_dates)} days)")
        print(f"  Test:  {fold.test_dates[0].date()} to {fold.test_dates[-1].date()} ({len(fold.test_dates)} days)")
    
    # Visualize
    cv.visualize_splits(data, expanding=True)
    print("\\n✓ Visualization saved to cv_splits.png")
\`\`\`

### 2. Purging and Embargo

**Purging**: Remove training samples after the training cutoff that could contain information about the test period due to overlapping positions.

**Embargo**: Add a gap before the test period to prevent information leakage from correlated events.

\`\`\`python
def apply_purging_embargo(
    train_end_date: datetime,
    test_start_date: datetime,
    positions: pd.DataFrame,
    purge_days: int = 5,
    embargo_days: int = 2
) -> Tuple[datetime, datetime]:
    """
    Apply purging and embargo to prevent leakage
    
    Args:
        train_end_date: End of training period
        test_start_date: Start of test period
        positions: DataFrame with position entry/exit dates
        purge_days: Days to purge after training
        embargo_days: Days to embargo before test
        
    Returns:
        (adjusted_train_end, adjusted_test_start)
    """
    # Find positions that span train_end_date
    spanning_positions = positions[
        (positions['entry_date'] <= train_end_date) &
        (positions['exit_date'] > train_end_date)
    ]
    
    if len(spanning_positions) > 0:
        # Find latest exit date of spanning positions
        latest_exit = spanning_positions['exit_date'].max()
        
        # Purge: move train end back to exclude these positions
        adjusted_train_end = min(
            train_end_date - timedelta(days=purge_days),
            latest_exit - timedelta(days=1)
        )
    else:
        adjusted_train_end = train_end_date - timedelta(days=purge_days)
    
    # Embargo: move test start forward
    adjusted_test_start = test_start_date + timedelta(days=embargo_days)
    
    return adjusted_train_end, adjusted_test_start
\`\`\`

### 3. Combinatorial Purged Cross-Validation (CPCV)

**Advanced technique** from López de Prado that generates all possible train/test combinations while respecting temporal ordering.

\`\`\`python
from itertools import combinations

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation
    
    Generates all valid train/test combinations
    """
    
    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        purge_gap: int = 5,
        embargo_gap: int = 2
    ):
        """
        Initialize CPCV
        
        Args:
            n_splits: Number of total splits
            n_test_splits: Number of splits used for testing in each fold
            purge_gap: Days to purge
            embargo_gap: Days to embargo
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
    
    def split(
        self,
        data: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorial splits
        
        Yields:
            (train_indices, test_indices) tuples
        """
        n = len(data)
        split_size = n // self.n_splits
        
        # Create split boundaries
        splits = []
        for i in range(self.n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.n_splits - 1 else n
            splits.append((start, end))
        
        # Generate all combinations of test splits
        for test_split_combo in combinations(range(self.n_splits), self.n_test_splits):
            # Collect test indices
            test_idx = []
            for split_num in test_split_combo:
                start, end = splits[split_num]
                test_idx.extend(range(start, end))
            
            # Collect train indices (excluding test splits and adjacent splits for purging)
            train_idx = []
            for split_num in range(self.n_splits):
                if split_num in test_split_combo:
                    continue  # Skip test splits
                
                # Check if adjacent to test split (for purging)
                is_adjacent = any(
                    abs(split_num - test_num) == 1
                    for test_num in test_split_combo
                )
                
                if is_adjacent:
                    continue  # Skip adjacent splits (purging)
                
                start, end = splits[split_num]
                train_idx.extend(range(start, end))
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            yield np.array(train_idx), np.array(test_idx)


# Example: Compare standard vs time-series CV
def compare_cv_methods(data: pd.DataFrame, strategy_func: callable):
    """Compare different CV methods"""
    from sklearn.model_selection import KFold
    
    results = {}
    
    # 1. Standard K-Fold (WRONG for time series)
    print("\\n1. Standard K-Fold (INCORRECT for time series):")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kfold.split(data):
        score = strategy_func(data.iloc[train_idx], data.iloc[test_idx])
        scores.append(score)
    results['kfold'] = np.mean(scores)
    print(f"   Mean Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print("   ⚠️  BIASED - Future data in training set!")
    
    # 2. Time Series CV (CORRECT)
    print("\\n2. Time Series CV with Purging/Embargo:")
    ts_cv = TimeSeriesCV(n_splits=5, test_size=252, purge_gap=5, embargo_gap=2)
    scores = []
    for train_idx, test_idx in ts_cv.split(data):
        score = strategy_func(data.iloc[train_idx], data.iloc[test_idx])
        scores.append(score)
    results['ts_cv'] = np.mean(scores)
    print(f"   Mean Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print("   ✓ UNBIASED - Proper temporal ordering")
    
    # 3. Combinatorial Purged CV (ADVANCED)
    print("\\n3. Combinatorial Purged CV:")
    cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
    scores = []
    for train_idx, test_idx in cpcv.split(data):
        score = strategy_func(data.iloc[train_idx], data.iloc[test_idx])
        scores.append(score)
    results['cpcv'] = np.mean(scores)
    print(f"   Mean Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"   Generated {len(scores)} combinations")
    print("   ✓ UNBIASED - Maximum use of data")
    
    return results
\`\`\`

## Critical Considerations

### 1. **Never Shuffle Time Series Data**
- Shuffling destroys temporal dependencies
- Creates impossible look-ahead bias
- Inflates performance metrics artificially

### 2. **Always Use Expanding or Rolling Windows**
- Expanding: Train set grows, realistic for production
- Rolling: Fixed window, tests adaptability

### 3. **Purge Overlapping Positions**
- Positions spanning train/test boundary create leakage
- Purge enough days to close all positions

### 4. **Embargo Recent Data**
- Prevents leakage from autocorrelated returns
- Typically 1-5 days depending on holding period

### 5. **Report All Folds**
- Don't cherry-pick best fold
- Report mean and std across all folds
- High variation across folds = unstable strategy

## Production Checklist

- [ ] Using time-aware CV (no shuffling)
- [ ] Purging implemented (gap after training)
- [ ] Embargo implemented (gap before test)
- [ ] Minimum training size enforced
- [ ] All folds reported (no cherry-picking)
- [ ] Visualization generated for review
- [ ] Documentation of CV choices
`,
    },
  ],
};

export default timeSeriesCrossValidation;
