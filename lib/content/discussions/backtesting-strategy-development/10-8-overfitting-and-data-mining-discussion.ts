import { Content } from "@/lib/types";

const overfittingAndDataMiningDiscussion: Content = {
    title: "Overfitting and Data Mining - Discussion Questions",
    description: "Deep-dive discussion questions on overfitting detection, multiple testing correction, and building robust trading strategies",
    sections: [
        {
            title: "Discussion Questions",
            content: `
# Discussion Questions: Overfitting and Data Mining

## Question 1: Building an Overfitting Detection System for a Quant Fund

**Scenario**: You're the head of research at a quantitative hedge fund with 15 researchers who collectively develop and test hundreds of trading strategies per year. Currently, there's no systematic process to detect overfitting, leading to strategies that backtest beautifully but fail in production. In the past year, 8 out of 12 deployed strategies underperformed expectations within 6 months, costing the fund $15M in lost capital and opportunity costs.

Your task is to design and implement a comprehensive overfitting detection system that:
1. Automatically screens new strategies for overfitting before deployment
2. Provides quantitative metrics and red flags
3. Enforces best practices (out-of-sample lockout, multiple testing corrections)
4. Generates reports for the investment committee
5. Tracks deployed strategies for degradation

**Address**:
- What specific tests and metrics would you implement?
- How would you enforce discipline around out-of-sample data?
- What thresholds would trigger rejection or further review?
- How would you handle the cultural challenge of researchers wanting to "peek" at holdout data?

### Comprehensive Answer

This is a critical infrastructure project that requires both technical rigor and organizational discipline. Here's a comprehensive system design:

#### System Architecture

\`\`\`python
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import hashlib
import json

class OverfitStatus(Enum):
    """Strategy overfitting status"""
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"
    PENDING = "pending"

class TestResult(Enum):
    """Individual test results"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

@dataclass
class StrategySubmission:
    """Strategy submission for overfitting review"""
    strategy_id: str
    researcher_name: str
    submission_date: datetime
    strategy_description: str
    economic_rationale: str
    
    # Data usage
    training_period: Tuple[datetime, datetime]
    validation_period: Optional[Tuple[datetime, datetime]]
    oos_period: Tuple[datetime, datetime]
    
    # Strategy characteristics
    n_parameters: int
    parameter_values: Dict[str, float]
    n_combinations_tested: int
    n_trades: int
    
    # Performance metrics
    is_sharpe: float
    oos_sharpe: float
    is_max_dd: float
    oos_max_dd: float
    
    # Supporting evidence
    cross_validation_results: Optional[Dict] = None
    walk_forward_results: Optional[Dict] = None
    monte_carlo_results: Optional[Dict] = None

class OverfittingDetectionSystem:
    """
    Production overfitting detection system
    """
    
    def __init__(
        self,
        oos_data_vault: str,  # Secure storage for locked OOS data
        alert_recipients: List[str],
        approval_threshold: float = 0.7
    ):
        self.oos_data_vault = oos_data_vault
        self.alert_recipients = alert_recipients
        self.approval_threshold = approval_threshold
        
        # Test configuration
        self.tests = self._configure_tests()
        
        # Audit trail
        self.audit_log: List[Dict] = []
    
    def _configure_tests(self) -> Dict[str, Dict]:
        """Configure overfitting tests with thresholds"""
        return {
            'degradation': {
                'weight': 0.25,
                'thresholds': {
                    TestResult.PASS: 10,  # <10% degradation
                    TestResult.WARNING: 30,  # 10-30% degradation
                    TestResult.FAIL: 50  # >50% degradation
                }
            },
            'parameter_sensitivity': {
                'weight': 0.15,
                'thresholds': {
                    TestResult.PASS: 0.15,  # CV < 15%
                    TestResult.WARNING: 0.30,
                    TestResult.FAIL: 0.50
                }
            },
            'degrees_of_freedom': {
                'weight': 0.10,
                'thresholds': {
                    TestResult.PASS: 50,  # >50 obs per param
                    TestResult.WARNING: 30,
                    TestResult.FAIL: 20
                }
            },
            'multiple_testing': {
                'weight': 0.15,
                'thresholds': {
                    TestResult.PASS: 10,  # <10 combinations
                    TestResult.WARNING: 100,
                    TestResult.FAIL: 1000
                }
            },
            'oos_significance': {
                'weight': 0.20,
                'thresholds': {
                    TestResult.PASS: 1.0,  # OOS Sharpe > 1.0
                    TestResult.WARNING: 0.5,
                    TestResult.FAIL: 0.0
                }
            },
            'economic_rationale': {
                'weight': 0.15,
                'thresholds': {}  # Manual review
            }
        }
    
    def evaluate_strategy(
        self,
        submission: StrategySubmission
    ) -> Dict[str, Any]:
        """
        Comprehensive overfitting evaluation
        
        Returns:
            Complete evaluation report with recommendation
        """
        print(f"\\nEvaluating strategy: {submission.strategy_id}")
        print(f"Researcher: {submission.researcher_name}")
        print("="*80)
        
        # Run all tests
        test_results = {}
        
        # Test 1: Performance Degradation
        test_results['degradation'] = self._test_degradation(submission)
        
        # Test 2: Parameter Sensitivity
        test_results['parameter_sensitivity'] = self._test_parameter_sensitivity(submission)
        
        # Test 3: Degrees of Freedom
        test_results['degrees_of_freedom'] = self._test_degrees_of_freedom(submission)
        
        # Test 4: Multiple Testing
        test_results['multiple_testing'] = self._test_multiple_testing(submission)
        
        # Test 5: OOS Statistical Significance
        test_results['oos_significance'] = self._test_oos_significance(submission)
        
        # Test 6: Economic Rationale (requires manual review)
        test_results['economic_rationale'] = self._test_economic_rationale(submission)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(test_results)
        
        # Determine status
        status = self._determine_status(overall_score, test_results)
        
        # Generate report
        report = {
            'strategy_id': submission.strategy_id,
            'researcher': submission.researcher_name,
            'evaluation_date': datetime.now(),
            'test_results': test_results,
            'overall_score': overall_score,
            'status': status,
            'recommendation': self._generate_recommendation(status, test_results),
            'required_actions': self._generate_required_actions(status, test_results)
        }
        
        # Log to audit trail
        self._log_evaluation(report)
        
        # Send alerts if needed
        if status == OverfitStatus.REJECTED:
            self._send_alert(report, severity='HIGH')
        elif status == OverfitStatus.NEEDS_REVIEW:
            self._send_alert(report, severity='MEDIUM')
        
        return report
    
    def _test_degradation(self, submission: StrategySubmission) -> Dict:
        """Test in-sample vs out-of-sample degradation"""
        degradation_pct = (
            (submission.is_sharpe - submission.oos_sharpe) / 
            submission.is_sharpe * 100
        ) if submission.is_sharpe != 0 else 100
        
        # Determine result
        if degradation_pct < self.tests['degradation']['thresholds'][TestResult.PASS]:
            result = TestResult.PASS
        elif degradation_pct < self.tests['degradation']['thresholds'][TestResult.WARNING]:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'degradation_pct': degradation_pct,
            'is_sharpe': submission.is_sharpe,
            'oos_sharpe': submission.oos_sharpe,
            'interpretation': self._interpret_degradation(degradation_pct)
        }
    
    def _interpret_degradation(self, degradation_pct: float) -> str:
        """Interpret degradation results"""
        if degradation_pct < 0:
            return f"Strategy improved OOS ({-degradation_pct:.1f}%) - excellent sign"
        elif degradation_pct < 10:
            return f"Minimal degradation ({degradation_pct:.1f}%) - very robust"
        elif degradation_pct < 25:
            return f"Moderate degradation ({degradation_pct:.1f}%) - acceptable"
        elif degradation_pct < 50:
            return f"Significant degradation ({degradation_pct:.1f}%) - concerning"
        else:
            return f"Severe degradation ({degradation_pct:.1f}%) - likely overfit"
    
    def _test_parameter_sensitivity(self, submission: StrategySubmission) -> Dict:
        """Test parameter sensitivity"""
        # This would require running the strategy with perturbed parameters
        # Simplified: assume we have pre-computed sensitivity
        
        # Coefficient of variation (would be calculated from actual tests)
        cv = 0.20  # Placeholder
        
        if cv < self.tests['parameter_sensitivity']['thresholds'][TestResult.PASS]:
            result = TestResult.PASS
        elif cv < self.tests['parameter_sensitivity']['thresholds'][TestResult.WARNING]:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'coefficient_of_variation': cv,
            'interpretation': f"{'Robust' if cv < 0.15 else 'Sensitive'} to parameter changes"
        }
    
    def _test_degrees_of_freedom(self, submission: StrategySubmission) -> Dict:
        """Test degrees of freedom"""
        obs_per_param = submission.n_trades / submission.n_parameters
        
        if obs_per_param > self.tests['degrees_of_freedom']['thresholds'][TestResult.PASS]:
            result = TestResult.PASS
        elif obs_per_param > self.tests['degrees_of_freedom']['thresholds'][TestResult.WARNING]:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'obs_per_parameter': obs_per_param,
            'n_parameters': submission.n_parameters,
            'n_trades': submission.n_trades,
            'interpretation': f"{'Sufficient' if obs_per_param > 30 else 'Insufficient'} data for parameters"
        }
    
    def _test_multiple_testing(self, submission: StrategySubmission) -> Dict:
        """Test for multiple testing problem"""
        n_tests = submission.n_combinations_tested
        
        # Calculate probability of false positive
        prob_false_positive = 1 - (1 - 0.05) ** n_tests
        
        # Bonferroni correction
        corrected_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
        
        if n_tests < self.tests['multiple_testing']['thresholds'][TestResult.PASS]:
            result = TestResult.PASS
        elif n_tests < self.tests['multiple_testing']['thresholds'][TestResult.WARNING]:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'n_combinations_tested': n_tests,
            'prob_false_positive': prob_false_positive,
            'bonferroni_alpha': corrected_alpha,
            'interpretation': f"{'Low' if n_tests < 10 else 'High'} multiple testing risk"
        }
    
    def _test_oos_significance(self, submission: StrategySubmission) -> Dict:
        """Test out-of-sample statistical significance"""
        oos_sharpe = submission.oos_sharpe
        n_periods = submission.n_trades
        
        # T-statistic for Sharpe ratio
        t_stat = oos_sharpe * np.sqrt(n_periods) / np.sqrt(1 + 0.5 * oos_sharpe**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_periods - 1))
        
        if oos_sharpe > self.tests['oos_significance']['thresholds'][TestResult.PASS]:
            result = TestResult.PASS
        elif oos_sharpe > self.tests['oos_significance']['thresholds'][TestResult.WARNING]:
            result = TestResult.WARNING
        else:
            result = TestResult.FAIL
        
        return {
            'result': result,
            'oos_sharpe': oos_sharpe,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'interpretation': f"{'Statistically significant' if p_value < 0.05 else 'Not significant'} edge"
        }
    
    def _test_economic_rationale(self, submission: StrategySubmission) -> Dict:
        """Test economic rationale (requires manual review)"""
        # This is a manual checklist
        checklist = {
            'has_clear_hypothesis': None,  # To be filled manually
            'supported_by_research': None,
            'makes_intuitive_sense': None,
            'not_purely_data_mined': None
        }
        
        return {
            'result': TestResult.WARNING,  # Always needs review
            'checklist': checklist,
            'rationale_provided': submission.economic_rationale,
            'interpretation': "Manual review required"
        }
    
    def _calculate_overall_score(self, test_results: Dict) -> float:
        """Calculate weighted overall score"""
        score = 0.0
        
        for test_name, test_result in test_results.items():
            if test_name not in self.tests:
                continue
            
            weight = self.tests[test_name]['weight']
            
            # Convert result to score
            result = test_result['result']
            if result == TestResult.PASS:
                test_score = 1.0
            elif result == TestResult.WARNING:
                test_score = 0.5
            else:  # FAIL
                test_score = 0.0
            
            score += weight * test_score
        
        return score
    
    def _determine_status(
        self,
        overall_score: float,
        test_results: Dict
    ) -> OverfitStatus:
        """Determine approval status"""
        # Check for critical failures
        critical_tests = ['degradation', 'oos_significance']
        has_critical_failure = any(
            test_results[test]['result'] == TestResult.FAIL
            for test in critical_tests
            if test in test_results
        )
        
        if has_critical_failure:
            return OverfitStatus.REJECTED
        elif overall_score >= self.approval_threshold:
            return OverfitStatus.APPROVED
        else:
            return OverfitStatus.NEEDS_REVIEW
    
    def _generate_recommendation(
        self,
        status: OverfitStatus,
        test_results: Dict
    ) -> str:
        """Generate recommendation text"""
        if status == OverfitStatus.APPROVED:
            return (
                "✓ APPROVED FOR DEPLOYMENT\\n"
                "Strategy passes all overfitting tests. "
                "Recommend paper trading for 1 month before live deployment."
            )
        elif status == OverfitStatus.NEEDS_REVIEW:
            return (
                "⚠ NEEDS REVIEW\\n"
                "Strategy shows some concerning patterns. "
                "Investment committee review required before deployment."
            )
        else:  # REJECTED
            failed_tests = [
                name for name, result in test_results.items()
                if result['result'] == TestResult.FAIL
            ]
            return (
                f"✗ REJECTED\\n"
                f"Strategy failed critical tests: {', '.join(failed_tests)}. "
                f"Do not deploy. Recommend redesign with focus on robustness."
            )
    
    def _generate_required_actions(
        self,
        status: OverfitStatus,
        test_results: Dict
    ) -> List[str]:
        """Generate required actions"""
        actions = []
        
        if status == OverfitStatus.APPROVED:
            actions.append("Proceed to paper trading")
            actions.append("Monitor for 30 days before live deployment")
            actions.append("Set up real-time performance tracking")
        
        elif status == OverfitStatus.NEEDS_REVIEW:
            # Specific actions based on warnings
            if test_results['degradation']['result'] == TestResult.WARNING:
                actions.append("Investigate cause of performance degradation")
            
            if test_results['parameter_sensitivity']['result'] == TestResult.WARNING:
                actions.append("Test robustness with parameter perturbations")
            
            if test_results['multiple_testing']['result'] == TestResult.WARNING:
                actions.append("Apply multiple testing corrections")
            
            actions.append("Present to investment committee")
        
        else:  # REJECTED
            actions.append("Do NOT deploy strategy")
            actions.append("Conduct root cause analysis")
            actions.append("Consider fundamental redesign")
            actions.append("Get more data or reduce parameters")
        
        return actions
    
    def _log_evaluation(self, report: Dict):
        """Log evaluation to audit trail"""
        self.audit_log.append({
            'timestamp': datetime.now(),
            'strategy_id': report['strategy_id'],
            'researcher': report['researcher'],
            'status': report['status'].value,
            'overall_score': report['overall_score']
        })
    
    def _send_alert(self, report: Dict, severity: str):
        """Send alert to recipients"""
        print(f"\\n{'='*80}")
        print(f"ALERT [{severity}]: Strategy {report['strategy_id']}")
        print(f"Status: {report['status'].value}")
        print(f"Researcher: {report['researcher']}")
        print(f"Recommendation: {report['recommendation']}")
        print(f"{'='*80}\\n")


class OOSDataVault:
    """
    Secure vault for out-of-sample data
    
    Enforces lockout and tracks access
    """
    
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.access_log: List[Dict] = []
        self.locked_datasets: Dict[str, Dict] = {}
    
    def lock_dataset(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        lockout_date: datetime,
        authorized_users: List[str]
    ) -> str:
        """
        Lock dataset for out-of-sample use
        
        Returns:
            Dataset hash for verification
        """
        # Create hash for integrity
        data_hash = hashlib.sha256(
            data.to_json().encode()
        ).hexdigest()
        
        self.locked_datasets[dataset_id] = {
            'lockout_date': lockout_date,
            'data_hash': data_hash,
            'authorized_users': authorized_users,
            'access_count': 0,
            'last_access': None
        }
        
        print(f"\\n🔒 Dataset {dataset_id} LOCKED")
        print(f"Lockout date: {lockout_date}")
        print(f"Hash: {data_hash[:16]}...")
        print(f"Authorized users: {', '.join(authorized_users)}")
        print(f"⚠️  This data is now FORBIDDEN for strategy development!")
        
        return data_hash
    
    def request_oos_data(
        self,
        dataset_id: str,
        user: str,
        purpose: str
    ) -> Optional[pd.DataFrame]:
        """
        Request access to OOS data (only for final validation)
        
        Access is logged and limited
        """
        if dataset_id not in self.locked_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = self.locked_datasets[dataset_id]
        
        # Check authorization
        if user not in dataset_info['authorized_users']:
            print(f"❌ ACCESS DENIED: {user} not authorized")
            return None
        
        # Check if already accessed too many times
        if dataset_info['access_count'] >= 1:
            print(f"❌ ACCESS DENIED: Dataset already accessed {dataset_info['access_count']} time(s)")
            print(f"   Last access: {dataset_info['last_access']}")
            return None
        
        # Log access
        self.access_log.append({
            'dataset_id': dataset_id,
            'user': user,
            'purpose': purpose,
            'timestamp': datetime.now()
        })
        
        dataset_info['access_count'] += 1
        dataset_info['last_access'] = datetime.now()
        
        print(f"\\n✓ OOS DATA ACCESS GRANTED")
        print(f"User: {user}")
        print(f"Purpose: {purpose}")
        print(f"⚠️  This access has been logged. Use responsibly!")
        
        # In production, would actually return the data
        return None  # Placeholder
    
    def audit_report(self) -> pd.DataFrame:
        """Generate audit report of all OOS data access"""
        return pd.DataFrame(self.access_log)


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = OverfittingDetectionSystem(
        oos_data_vault="/secure/oos_data",
        alert_recipients=["risk@fund.com", "cio@fund.com"],
        approval_threshold=0.7
    )
    
    # Example submission
    submission = StrategySubmission(
        strategy_id="MOMENTUM_v3.2",
        researcher_name="Jane Smith",
        submission_date=datetime.now(),
        strategy_description="Momentum strategy with adaptive lookback",
        economic_rationale="Exploits slow information diffusion in mid-cap stocks",
        training_period=(datetime(2018, 1, 1), datetime(2022, 12, 31)),
        validation_period=(datetime(2021, 1, 1), datetime(2022, 12, 31)),
        oos_period=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
        n_parameters=4,
        parameter_values={'lookback': 30, 'threshold': 2.0},
        n_combinations_tested=50,
        n_trades=145,
        is_sharpe=1.8,
        oos_sharpe=1.5,
        is_max_dd=-0.12,
        oos_max_dd=-0.15
    )
    
    # Evaluate
    report = system.evaluate_strategy(submission)
    
    # Print summary
    print("\\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Strategy: {report['strategy_id']}")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print(f"Status: {report['status'].value.upper()}")
    print(f"\\n{report['recommendation']}")
    print(f"\\nRequired Actions:")
    for action in report['required_actions']:
        print(f"  • {action}")
\`\`\`

#### Key Implementation Points

**1. Automated Testing Suite**:
- 6 quantitative tests with weighted scoring
- Clear pass/warning/fail thresholds
- Automatic recommendation generation

**2. OOS Data Vault**:
- Cryptographic hashing for integrity
- Access logging and limits (max 1 access per dataset)
- Authorization controls
- Audit trail

**3. Cultural Enforcement**:
- Make OOS access deliberately difficult
- Log every access with justification
- Require CIO approval for OOS data unlocking
- Public shame wall for premature OOS peeking
- Bonus clawbacks for researchers who deploy overfit strategies

**4. Reporting**:
- Automated reports to investment committee
- Traffic light system (green/yellow/red)
- Required actions clearly stated
- Historical tracking of all evaluations

**5. Thresholds (based on industry standards)**:
- Degradation: <10% pass, 10-30% warning, >50% fail
- Observations per parameter: >50 pass, 30-50 warning, <20 fail
- Multiple tests: <10 combinations pass, 100-1000 warning, >1000 fail
- OOS Sharpe: >1.0 pass, 0.5-1.0 warning, <0.5 fail

This system would prevent the vast majority of overfit strategies from reaching production while maintaining research velocity.

---

## Question 2: The "Golden Strategy" Problem - When Everything Looks Perfect

**Scenario**: A researcher comes to you excited about a new strategy they've discovered. The backtest results are incredible:
- Sharpe ratio: 4.2 (in-sample), 3.8 (out-of-sample)
- Max drawdown: -5%
- Win rate: 78%
- Works across multiple asset classes
- Only 3 parameters
- "Simple" moving average crossover with a twist

This seems too good to be true. The researcher insists they didn't data mine—they had the idea from a research paper and it just happened to work. They've followed all protocols: locked OOS data, proper train/test split, cross-validation shows consistency.

**Task**: As the head of risk, you're skeptical. What additional checks would you perform? What are the possible explanations for such exceptional performance? How would you determine if this is:
(a) A genuine breakthrough
(b) Subtle overfitting that passed initial checks
(c) Data quality issues or biases
(d) Implementation bugs

### Comprehensive Answer

Exceptional results demand exceptional scrutiny. Here's how to investigate:

#### Systematic Investigation Framework

\`\`\`python
class GoldenStrategyInvestigator:
    """
    Investigate strategies that seem too good to be true
    """
    
    def __init__(self, strategy_results: Dict):
        self.results = strategy_results
        self.red_flags: List[str] = []
        self.possible_explanations: List[str] = []
    
    def investigate(self) -> Dict[str, Any]:
        """
        Comprehensive investigation
        
        Returns:
            Investigation report
        """
        print("\\n" + "="*80)
        print("EXCEPTIONAL PERFORMANCE INVESTIGATION")
        print("="*80)
        print(f"Sharpe Ratio: {self.results['sharpe']}")
        print(f"Max Drawdown: {self.results['max_dd']}")
        print(f"Win Rate: {self.results['win_rate']:.1%}")
        print("")
        
        # Run all checks
        self._check_data_quality()
        self._check_look_ahead_bias()
        self._check_survivorship_bias()
        self._check_implementation_bugs()
        self._check_statistical_likelihood()
        self._check_regime_dependency()
        self._check_hidden_complexity()
        self._check_correlation_with_known_factors()
        
        # Generate conclusion
        conclusion = self._generate_conclusion()
        
        return {
            'red_flags': self.red_flags,
            'possible_explanations': self.possible_explanations,
            'conclusion': conclusion,
            'recommendation': self._generate_recommendation()
        }
    
    def _check_data_quality(self):
        """Check for data quality issues"""
        checks = [
            "Are all prices adjusted for splits/dividends?",
            "Is data point-in-time (no look-ahead)?",
            "Are there any suspicious gaps or outliers?",
            "Is bid-ask spread data included?",
            "Are delisted stocks included (survivorship bias)?",
            "Is trading volume realistic for position sizes?"
        ]
        
        print("Data Quality Checks:")
        for check in checks:
            print(f"  □ {check}")
        print("")
        
        self.possible_explanations.append(
            "Data quality issue: Unadjusted prices, missing delisted stocks, "
            "or incomplete corporate actions"
        )
    
    def _check_look_ahead_bias(self):
        """Check for subtle look-ahead bias"""
        print("Look-Ahead Bias Checks:")
        print("  □ Are indicators calculated before signal generation?")
        print("  □ Is there any use of 'next day's open' before it would be known?")
        print("  □ Are stop losses placed at prices that weren't available?")
        print("  □ Is any future information leaking into past decisions?")
        print("")
        
        self.red_flags.append("Verify no look-ahead bias in signal generation")
    
    def _check_survivorship_bias(self):
        """Check for survivorship bias"""
        print("Survivorship Bias Checks:")
        print("  □ Strategy tested on current index constituents only?")
        print("  □ Are delisted/bankrupt companies included?")
        print("  □ Are IPOs excluded from early period?")
        print("")
        
        sharpe = self.results['sharpe']
        if sharpe > 3.0:
            self.red_flags.append(
                "Sharpe > 3.0: Possible survivorship bias if using current index members"
            )
            self.possible_explanations.append(
                "Survivorship bias: Strategy only tested on stocks that survived, "
                "excluding bankruptcies and delistings"
            )
    
    def _check_implementation_bugs(self):
        """Check for implementation bugs"""
        print("Implementation Bug Checks:")
        print("  □ Are positions sized correctly?")
        print("  □ Are costs (commission + slippage) included?")
        print("  □ Is leverage calculated correctly?")
        print("  □ Are signals generated before trades executed?")
        print("  □ Is there any off-by-one error in indexing?")
        print("")
        
        self.possible_explanations.append(
            "Implementation bug: Off-by-one error allowing perfect foresight, "
            "or missing transaction costs"
        )
    
    def _check_statistical_likelihood(self):
        """Check statistical likelihood of results"""
        sharpe = self.results['sharpe']
        n_years = self.results.get('n_years', 5)
        
        # Under null hypothesis (no edge), what's probability of this Sharpe?
        # Sharpe ~ N(0, sqrt(1/T)) under null
        se = np.sqrt(1 / (n_years * 252))
        z_score = sharpe / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print(f"Statistical Likelihood:")
        print(f"  Z-score: {z_score:.2f}")
        print(f"  P-value: {p_value:.2e}")
        print(f"  Interpretation: {'Extremely unlikely by chance' if p_value < 0.0001 else 'Possible by chance'}")
        print("")
        
        if sharpe > 4.0:
            self.red_flags.append(
                f"Sharpe {sharpe:.1f} is extraordinarily high. "
                f"Less than 0.01% of strategies achieve this legitimately."
            )
    
    def _check_regime_dependency(self):
        """Check if performance is regime-dependent"""
        print("Regime Dependency Checks:")
        print("  □ Does strategy work in both bull and bear markets?")
        print("  □ Performance during 2008 crisis?")
        print("  □ Performance during 2020 COVID crash?")
        print("  □ Performance in rising/falling rate environments?")
        print("")
        
        self.possible_explanations.append(
            "Regime dependency: Strategy only works in specific market conditions "
            "that happened to dominate the backtest period"
        )
    
    def _check_hidden_complexity(self):
        """Check for hidden complexity"""
        n_params = self.results.get('n_parameters', 3)
        
        print(f"Complexity Checks (stated: {n_params} parameters):")
        print("  □ Are there implicit parameters in 'magic numbers'?")
        print("  □ Are asset selection rules parameter-free?")
        print("  □ Are there conditional logic branches?")
        print("  □ Is the 'twist' actually adding 10 hidden parameters?")
        print("")
        
        if n_params < 5 and sharpe > 3.0:
            self.red_flags.append(
                "Suspiciously high Sharpe with few parameters. "
                "Check for hidden complexity."
            )
    
    def _check_correlation_with_known_factors(self):
        """Check if strategy is just a known factor in disguise"""
        print("Factor Exposure Checks:")
        print("  □ Calculate correlation with: Market, Size, Value, Momentum")
        print("  □ Check for hidden sector bets")
        print("  □ Verify not just leveraged market exposure")
        print("  □ Compare to known anomalies/strategies")
        print("")
        
        self.possible_explanations.append(
            "Factor exposure: Strategy is actually just a leveraged bet on "
            "a known factor (e.g., 3x momentum) made to look novel"
        )
    
    def _generate_conclusion(self) -> str:
        """Generate investigation conclusion"""
        if len(self.red_flags) == 0:
            return "No major red flags found, but remain vigilant."
        elif len(self.red_flags) <= 2:
            return "Some concerns identified. Recommend additional validation."
        else:
            return "Multiple red flags. High probability of error or bias."
    
    def _generate_recommendation(self) -> str:
        """Generate recommendation"""
        sharpe = self.results['sharpe']
        
        if sharpe > 4.0:
            return (
                "RECOMMEND: Independent verification by separate team. "
                "Code review by senior developer. "
                "Extended paper trading (6+ months) before any capital deployment. "
                "Start with minimal allocation if approved."
            )
        else:
            return (
                "RECOMMEND: Standard validation process with extra scrutiny. "
                "Paper trading for 3 months minimum."
            )


# Example investigation
if __name__ == "__main__":
    results = {
        'sharpe': 4.2,
        'max_dd': -0.05,
        'win_rate': 0.78,
        'n_parameters': 3,
        'n_years': 5
    }
    
    investigator = GoldenStrategyInvestigator(results)
    report = investigator.investigate()
    
    print("\\nINVESTIGATION SUMMARY")
    print("="*80)
    print(f"Red Flags: {len(report['red_flags'])}")
    for flag in report['red_flags']:
        print(f"  ⚠️  {flag}")
    
    print(f"\\nPossible Explanations:")
    for explanation in report['possible_explanations']:
        print(f"  • {explanation}")
    
    print(f"\\nConclusion: {report['conclusion']}")
    print(f"\\nRecommendation: {report['recommendation']}")
\`\`\`

#### Most Likely Explanations (Ranked)

1. **Survivorship Bias** (40% probability)
   - Only testing current index members
   - Excludes delisted/bankrupt stocks
   - Easy to miss, produces exactly this pattern

2. **Look-Ahead Bias** (30% probability)
   - Subtle off-by-one error
   - Using "open" price before it's available
   - Signal generated after the fact

3. **Missing Transaction Costs** (15% probability)
   - No slippage modeled
   - Unrealistic fill assumptions
   - Ignoring market impact

4. **Regime Luck** (10% probability)
   - Works only in specific conditions
   - Backtest period was favorable
   - Won't generalize forward

5. **Genuine Discovery** (5% probability)
   - Possible but unlikely
   - Would need exceptional explanation

#### Definitive Tests

**Test 1: Independent Replication**

\`\`\`python
def independent_replication_test(strategy_code: str) -> bool:
    """Have separate team replicate from description"""
    # Different team, different codebase
    # If they can't replicate results, something is wrong
    pass
\`\`\`

**Test 2: Forward Test on New Data**

\`\`\`python
def forward_test(strategy, new_data) -> Dict:
    """Test on truly unseen data (recent months)"""
    # If performance collapses, likely overfit
    pass
\`\`\`

**Test 3: Stress Test Historical Crises**
```python
def crisis_test(strategy) -> Dict:
        """How did it perform in 2008, 2020, etc?"""
    # Real strategies should survive(or explain why not)
    pass
            ```

#### Recommended Actions

1. **Immediate**: Code review by independent senior developer
2. **Short-term**: Paper trade for 6 months with real capital allocation rules
3. **Medium-term**: If paper trading succeeds, start with 1% of target allocation
4. **Long-term**: Graduate to full allocation only after 12 months of live validation

**Golden Rule**: When something seems too good to be true in trading, it almost always is. The burden of proof is on the strategy to prove it's legitimate, not on the skeptics to find the bug.

---

## Question 3: Regulatory Compliance and Overfitting

**Scenario**: Your fund is being audited by the SEC, and they're questioning whether your backtest methodology allows for adequate evaluation of strategy robustness. They're concerned that you're selling strategies to clients based on backtest results that may be overfit. Specifically, they want to see:

1. Documentation that out-of-sample data was truly held out
2. Evidence of multiple testing corrections
3. Procedures to prevent data mining
4. Disclosure of how many strategies were tested vs deployed

Design a compliance framework that satisfies regulators while still allowing productive research.

### Comprehensive Answer

Regulatory compliance for backtesting requires rigorous documentation and processes that prove strategies weren't cherry-picked from hundreds of failed attempts.

**Framework**: See comprehensive compliance system in the code above, with audit trails, access controls, and mandatory disclosures.

**Key Regulatory Requirements**:
1. **Documented lockout procedures** for OOS data
2. **Complete audit trail** of all strategy tests
3. **Multiple testing disclosures** in marketing materials  
4. **Separation of research and production** teams
5. **Independent validation** before client money

This protects both the fund and its clients from the dangers of overfit strategies.
`,
    },
  ],
};

export default overfittingAndDataMiningDiscussion;

