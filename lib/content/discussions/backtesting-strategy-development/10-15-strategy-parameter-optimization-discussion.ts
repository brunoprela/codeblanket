import { Content } from '@/lib/types';

const strategyParameterOptimizationDiscussion: Content = {
  title: 'Strategy Parameter Optimization - Discussion Questions',
  description:
    'Deep-dive discussion questions on parameter optimization, overfitting prevention, and production deployment',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Strategy Parameter Optimization

## Question 1: The Optimization Paradox

**Scenario**: Your team has developed a mean-reversion strategy with 8 parameters. After extensive optimization using Bayesian methods across 3 years of data, you've found a parameter set with a Sharpe ratio of 2.8.

However, your senior quant raises concerns: "The more thoroughly we optimize, the more likely we are to overfit. But if we don't optimize, we're leaving performance on the table. How do we find the right balance?"

**Questions:**1. How do you determine if your optimization process has gone too far?
2. What safeguards prevent overfitting during optimization?
3. How do you balance optimization depth with generalization?

### Comprehensive Answer

This is the fundamental tension in quantitative trading: optimization is necessary but dangerous.

**Detecting Over-Optimization:**

\`\`\`python
class OptimizationValidator:
    """
    Validate optimization results to detect overfitting
    """
    
    def __init__(self, optimization_results: Dict):
        self.results = optimization_results
    
    def check_parameter_stability(
        self,
        optimal_params: Dict,
        performance_threshold: float = 0.95
    ) -> Dict:
        """
        Test if small parameter changes dramatically affect performance
        
        Overfit strategies are hypersensitive to parameters
        """
        sensitivities = {}
        
        for param_name, optimal_value in optimal_params.items():
            # Test ±10% variations
            variations = [
                optimal_value * 0.9,
                optimal_value,
                optimal_value * 1.1
            ]
            
            performance_scores = []
            for variation in variations:
                modified_params = optimal_params.copy()
                modified_params[param_name] = variation
                score = self._evaluate_params(modified_params)
                performance_scores.append(score)
            
            # Calculate sensitivity
            pct_changes = [
                abs(score - performance_scores[1]) / performance_scores[1]
                for score in performance_scores if score != performance_scores[1]
            ]
            
            avg_sensitivity = np.mean(pct_changes)
            sensitivities[param_name] = avg_sensitivity
        
        # Interpretation
        max_sensitivity = max(sensitivities.values())
        
        if max_sensitivity > 0.20:  # >20% performance drop for 10% param change
            status = "UNSTABLE"
            interpretation = (
                "Parameters highly sensitive. Strategy likely overfit. "
                "Small parameter changes cause large performance swings."
            )
        elif max_sensitivity > 0.10:
            status = "MODERATE"
            interpretation = "Some sensitivity detected. Monitor carefully."
        else:
            status = "STABLE"
            interpretation = "Parameters robust to small variations."
        
        return {
            'sensitivities': sensitivities,
            'max_sensitivity': max_sensitivity,
            'status': status,
            'interpretation': interpretation
        }
    
    def check_curve_shape(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze optimization landscape
        
        Smooth curves = likely robust
        Sharp peaks = likely overfit
        """
        # For 1D parameter sweep
        if len(results_df.columns) == 2:  # param + score
            scores = results_df.iloc[:, 1].values
            
            # Calculate smoothness (second derivative)
            second_derivative = np.diff(np.diff(scores))
            smoothness = np.std(second_derivative)
            
            if smoothness < 0.1:
                return {
                    'landscape': 'SMOOTH',
                    'interpretation': 'Gradual performance changes suggest robust parameters'
                }
            else:
                return {
                    'landscape': 'JAGGED',
                    'interpretation': 'Sharp peaks suggest overfitting to noise'
                }
        
        return {'landscape': 'UNKNOWN'}
    
    def check_in_sample_out_sample_correlation(
        self,
        in_sample_rankings: List[int],
        out_sample_rankings: List[int]
    ) -> Dict:
        """
        Check if parameters that work well in-sample also work well out-of-sample
        
        High correlation = good
        Low/negative correlation = overfitting
        """
        from scipy.stats import spearmanr
        
        correlation, p_value = spearmanr(in_sample_rankings, out_sample_rankings)
        
        if correlation > 0.5 and p_value < 0.05:
            status = "GOOD"
            interpretation = "Strong correlation between IS and OOS rankings. Parameters generalize well."
        elif correlation > 0.2:
            status = "MODERATE"
            interpretation = "Weak correlation. Some parameters may be overfit."
        else:
            status = "POOR"
            interpretation = "Low/negative correlation. Severe overfitting. Parameters don't generalize."
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'status': status,
            'interpretation': interpretation
        }


def optimization_best_practices():
    """
    Best practices for robust optimization
    """
    
    practices = {
        '1. Walk-Forward Optimization': {
            'description': 'Optimize on window, test on next window, roll forward',
            'prevents': 'Single-period overfitting',
            'implementation': 'Use WalkForwardOptimizer class'
        },
        
        '2. Limit Parameter Space': {
            'description': 'Reduce number of parameters and their ranges',
            'prevents': 'Excessive degrees of freedom',
            'rule_of_thumb': 'Need 10x data points per parameter'
        },
        
        '3. Penalize Complexity': {
            'description': 'Add complexity penalty to objective function',
            'prevents': 'Overly complex parameter sets',
            'example': 'Score = Sharpe - 0.1 * num_parameters'
        },
        
        '4. Use Multiple Metrics': {
            'description': 'Optimize on Sharpe, but check drawdown, win rate, etc.',
            'prevents': 'Gaming single metric',
            'example': 'Require Sharpe > 1.5 AND max_dd < 15%'
        },
        
        '5. Parameter Averaging': {
            'description': 'Average optimal parameters across multiple periods',
            'prevents': 'Period-specific overfitting',
            'implementation': 'Use median of walk-forward results'
        },
        
        '6. Minimum Performance Threshold': {
            'description': 'Reject if OOS < 70% of in-sample',
            'prevents': 'Accepting overfit strategies',
            'example': 'If IS Sharpe = 2.0, require OOS Sharpe >= 1.4'
        },
        
        '7. Regularization': {
            'description': 'Shrink parameters toward simple baselines',
            'prevents': 'Extreme parameter values',
            'example': 'L1/L2 penalties on parameter deviations'
        },
        
        '8. Early Stopping': {
            'description': 'Stop optimization before perfect fit',
            'prevents': 'Optimization running too long',
            'implementation': 'Stop when validation score plateaus'
        }
    }
    
    print("\\nOPTIMIZATION BEST PRACTICES")
    print("="*80)
    for name, details in practices.items():
        print(f"\\n{name}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    print("\\n" + "="*80)


if __name__ == "__main__":
    optimization_best_practices()
\`\`\`

**Recommended Approach:**1. **Phase 1: Coarse Search (Grid/Random)**
   - Wide parameter ranges
   - Identify promising regions
   - Fast, prevents local minima

2. **Phase 2: Fine-Tuning (Bayesian)**
   - Narrow ranges around promising regions
   - More evaluations
   - Find local optimum

3. **Phase 3: Walk-Forward Validation**
   - Test parameters across multiple periods
   - Average results
   - Validate robustness

4. **Phase 4: Stability Testing**
   - Parameter sensitivity analysis
   - Multiple metric evaluation
   - Final go/no-go decision

---

## Question 2: Real-Time Parameter Adaptation

**Scenario**: Your strategy has been live for 6 months. Performance has degraded from Sharpe 1.8 (backtest) to 1.0 (live). Your team debates two approaches:

**Approach A**: Re-optimize parameters monthly on rolling 2-year window
- Pros: Adapts to changing markets
- Cons: Risk of overfitting to recent data

**Approach B**: Keep original parameters, only change if performance drops below threshold
- Pros: Avoids overfitting
- Cons: May miss regime changes

**Design a systematic approach to parameter adaptation in live trading.**

### Comprehensive Answer

\`\`\`python
class AdaptiveParameterManager:
    """
    Manage parameter adaptation in live trading
    
    Balance adaptation vs stability
    """
    
    def __init__(
        self,
        strategy_name: str,
        initial_params: Dict,
        performance_threshold: float = 0.7
    ):
        self.strategy_name = strategy_name
        self.current_params = initial_params
        self.param_history = [(datetime.now(), initial_params)]
        self.performance_threshold = performance_threshold
        
        # Tracking
        self.reoptimization_dates = []
        self.performance_log = []
    
    def should_reoptimize(
        self,
        current_performance: Dict,
        baseline_performance: Dict
    ) -> Dict:
        """
        Decide if reoptimization is warranted
        
        Criteria:
        1. Performance degradation > threshold
        2. Minimum time elapsed (avoid overreacting)
        3. Statistical significance of degradation
        4. Regime change detected
        """
        reasons = []
        
        # Check 1: Performance degradation
        current_sharpe = current_performance['sharpe']
        baseline_sharpe = baseline_performance['sharpe']
        degradation = (baseline_sharpe - current_sharpe) / baseline_sharpe
        
        if degradation > (1 - self.performance_threshold):
            reasons.append(f"Performance degradation: {degradation:.1%}")
        
        # Check 2: Minimum time elapsed
        if len(self.reoptimization_dates) > 0:
            days_since_last = (datetime.now() - self.reoptimization_dates[-1]).days
            if days_since_last < 90:  # Minimum 3 months
                return {
                    'reoptimize': False,
                    'reason': f"Only {days_since_last} days since last reoptimization. Wait {90-days_since_last} more days."
                }
        
        # Check 3: Statistical significance
        # Use t-test on recent vs baseline returns
        # (Simplified - would use actual returns)
        
        # Check 4: Regime change
        # Detect if market characteristics changed
        # (Simplified - would use regime detection model)
        
        if reasons:
            return {
                'reoptimize': True,
                'reasons': reasons,
                'recommendation': 'Trigger reoptimization'
            }
        else:
            return {
                'reoptimize': False,
                'reason': 'Performance acceptable'
            }
    
    def adaptive_reoptimize(
        self,
        recent_data: pd.DataFrame,
        optimizer: 'Optimizer'
    ) -> Dict:
        """
        Adaptive reoptimization with safeguards
        
        1. Re-optimize on recent data
        2. Compare new vs old parameters
        3. Test both sets on validation period
        4. Only switch if new parameters clearly better
        """
        # Re-optimize
        new_params = optimizer.optimize()
        
        # Compare to current parameters
        comparison = self._compare_parameters(
            self.current_params,
            new_params,
            recent_data
        )
        
        # Decision rules
        if comparison['new_oos_performance'] > comparison['old_oos_performance'] * 1.10:
            # New parameters 10%+ better
            decision = 'SWITCH'
            self.current_params = new_params
            self.param_history.append((datetime.now(), new_params))
            self.reoptimization_dates.append(datetime.now())
        elif comparison['new_oos_performance'] > comparison['old_oos_performance']:
            # New parameters marginally better
            decision = 'BLEND'
            # Use weighted average of old and new
            blended_params = {
                key: 0.7 * self.current_params[key] + 0.3 * new_params[key]
                for key in self.current_params.keys()
            }
            self.current_params = blended_params
            self.param_history.append((datetime.now(), blended_params))
        else:
            # Keep current parameters
            decision = 'KEEP'
        
        return {
            'decision': decision,
            'old_params': self.current_params,
            'new_params': new_params,
            'comparison': comparison
        }
    
    def _compare_parameters(
        self,
        old_params: Dict,
        new_params: Dict,
        data: pd.DataFrame
    ) -> Dict:
        """Compare old vs new parameters on validation data"""
        
        # Split data: optimization period and validation period
        split_point = int(len(data) * 0.7)
        validation_data = data.iloc[split_point:]
        
        # Test both parameter sets
        old_performance = self._evaluate_params(old_params, validation_data)
        new_performance = self._evaluate_params(new_params, validation_data)
        
        return {
            'old_oos_performance': old_performance['sharpe'],
            'new_oos_performance': new_performance['sharpe'],
            'difference_pct': (
                (new_performance['sharpe'] - old_performance['sharpe']) /
                old_performance['sharpe'] * 100
            )
        }
    
    def _evaluate_params(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Evaluate parameters on data"""
        # Placeholder - would run backtest
        return {'sharpe': 1.5}


# Recommended approach: Hybrid
# - Monitor continuously
# - Reoptimize quarterly IF performance degrades
# - Use blended parameters (70% old, 30% new) for gradual adaptation
# - Maintain parameter history for analysis
\`\`\`

**Recommended Strategy: Conditional Adaptive**1. **Continuous Monitoring**: Track performance daily
2. **Trigger Thresholds**: Only reoptimize if:
   - Performance < 70% of baseline for 3+ months
   - Regime change detected
   - Minimum 3 months since last reoptimization
3. **Conservative Updates**: Blend old/new parameters (70/30)
4. **Validation Required**: New parameters must beat old on OOS data
5. **Rollback Plan**: Immediate reversion if new parameters perform poorly

This balances adaptation with stability.

---

## Question 3: Multi-Objective Optimization

**Most optimization focuses on a single metric (usually Sharpe ratio). But real trading requires balancing multiple objectives:**

- Maximize returns
- Minimize drawdowns
- Minimize transaction costs
- Maintain reasonable turnover
- Limit correlation to other strategies

**Design a multi-objective optimization framework that balances these competing goals.**

### Comprehensive Answer

\`\`\`python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ObjectiveScore:
    """Score for a single objective"""
    name: str
    value: float
    weight: float
    target: Optional[float] = None

class MultiObjectiveOptimizer:
    """
    Multi-objective parameter optimization
    
    Balances multiple competing objectives
    """
    
    def __init__(
        self,
        objectives: List[Dict],
        constraints: List[Dict] = None
    ):
        """
        Args:
            objectives: List of {name, weight, direction}
            constraints: List of {name, min, max}
        """
        self.objectives = objectives
        self.constraints = constraints or []
    
    def evaluate_multi_objective(
        self,
        metrics: Dict
    ) -> float:
        """
        Calculate composite score from multiple objectives
        
        Uses weighted sum approach
        """
        total_score = 0
        
        for objective in self.objectives:
            name = objective['name']
            weight = objective['weight']
            direction = objective.get('direction', 'maximize')  # or 'minimize'
            
            raw_value = metrics.get(name, 0)
            
            # Normalize to [0, 1]
            normalized = self._normalize_metric(name, raw_value)
            
            # Invert if minimizing
            if direction == 'minimize':
                normalized = 1 - normalized
            
            total_score += weight * normalized
        
        # Apply constraint penalties
        for constraint in self.constraints:
            name = constraint['name']
            value = metrics.get(name, 0)
            
            if 'min' in constraint and value < constraint['min']:
                # Penalty for violating minimum
                violation = (constraint['min'] - value) / constraint['min']
                total_score *= (1 - violation)  # Reduce score
            
            if 'max' in constraint and value > constraint['max']:
                # Penalty for violating maximum
                violation = (value - constraint['max']) / constraint['max']
                total_score *= (1 - violation)
        
        return total_score
    
    def _normalize_metric(self, name: str, value: float) -> float:
        """Normalize metric to [0, 1] range"""
        # Define typical ranges for common metrics
        ranges = {
            'sharpe': (0, 3),
            'total_return': (0, 1.0),
            'max_drawdown': (-0.5, 0),
            'turnover': (0, 100),
            'correlation': (0, 1)
        }
        
        if name in ranges:
            min_val, max_val = ranges[name]
            normalized = (value - min_val) / (max_val - min_val)
            return np.clip(normalized, 0, 1)
        
        return value


# Example: Optimize for multiple objectives
def example_multi_objective():
    """Example multi-objective optimization"""
    
    optimizer = MultiObjectiveOptimizer(
        objectives=[
            {'name': 'sharpe', 'weight': 0.40, 'direction': 'maximize'},
            {'name': 'max_drawdown', 'weight': 0.25, 'direction': 'minimize'},
            {'name': 'turnover', 'weight': 0.15, 'direction': 'minimize'},
            {'name': 'correlation_to_portfolio', 'weight': 0.20, 'direction': 'minimize'}
        ],
        constraints=[
            {'name': 'sharpe', 'min': 1.0},  # Minimum acceptable Sharpe
            {'name': 'max_drawdown', 'max': -0.20},  # Max 20% drawdown
            {'name': 'turnover', 'max': 50}  # Max 50x annual turnover
        ]
    )
    
    # Evaluate candidate strategy
    metrics = {
        'sharpe': 1.8,
        'max_drawdown': -0.12,
        'turnover': 30,
        'correlation_to_portfolio': 0.3
    }
    
    score = optimizer.evaluate_multi_objective(metrics)
    print(f"Multi-objective score: {score:.3f}")

if __name__ == "__main__":
    example_multi_objective()
\`\`\`

**Key Principle**: Don't optimize Sharpe alone. Balance risk, return, costs, and portfolio fit.
`,
    },
  ],
};

export default strategyParameterOptimizationDiscussion;
