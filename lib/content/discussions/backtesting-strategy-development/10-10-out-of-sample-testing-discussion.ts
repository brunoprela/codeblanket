import { Content } from '@/lib/types';

const outOfSampleTestingDiscussion: Content = {
  title: 'Out-of-Sample Testing - Discussion Questions',
  description:
    'Deep-dive discussion questions on proper OOS testing procedures, test set management, and transition to live trading',
  sections: [
    {
      title: 'Discussion Questions',
      content: `
# Discussion Questions: Out-of-Sample Testing

## Question 1: Designing a Test Set Management System

**Scenario**: Your quantitative hedge fund has experienced multiple incidents where test sets were accidentally compromised:
- A researcher "peeked" at test results and then modified their strategy
- Another researcher ran the same strategy on the test set multiple times with different parameters
- Test data was inadvertently included in a training dataset due to date indexing errors

These incidents cost the fund approximately $8M in capital deployed to overfit strategies. You've been tasked with designing a secure test set management system that makes it technologically impossible to compromise test data while still allowing legitimate final validation.

**Design a comprehensive system that addresses:**
1. How to physically/logically separate test data
2. Access control and audit logging
3. Cryptographic verification of data integrity
4. One-time-access enforcement
5. Mandatory reporting of results (good or bad)
6. Integration with existing research workflow

### Comprehensive Answer

A robust test set management system requires both technical controls and organizational process enforcement.

\`\`\`python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
import logging

class AccessLevel(Enum):
    """Access levels for data"""
    RESEARCHER = "researcher"  # Can access train/val
    VALIDATOR = "validator"  # Can access test (once)
    ADMIN = "admin"  # Can manage system

class DataType(Enum):
    """Types of data"""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"

@dataclass
class AccessLog:
    """Log entry for data access"""
    timestamp: datetime
    user: str
    data_type: DataType
    strategy_id: str
    access_granted: bool
    reason: str

class SecureTestSetVault:
    """
    Cryptographically secure vault for test data
    
    Features:
    - Encrypted storage
    - One-time access enforcement
    - Audit logging
    - Integrity verification
    """
    
    def __init__(
        self,
        vault_path: str,
        db_path: str,
        encryption_key: Optional[bytes] = None
    ):
        self.vault_path = Path(vault_path)
        self.db_path = db_path
        
        # Initialize encryption
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.cipher = Fernet(encryption_key)
        
        # Initialize database for access control
        self._init_database()
        
        # Setup logging
        logging.basicConfig(
            filename='test_set_access.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize access control database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                creation_date TEXT NOT NULL,
                data_hash TEXT NOT NULL,
                encrypted_path TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                max_access INTEGER DEFAULT 1,
                is_locked BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                access_granted BOOLEAN NOT NULL,
                reason TEXT,
                results_reported BOOLEAN DEFAULT 0,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                performance_metrics TEXT NOT NULL,
                decision TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def lock_dataset(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        dataset_id: str
    ) -> Dict:
        """
        Lock a dataset in the secure vault
        
        Args:
            data: DataFrame to lock
            data_type: Type of data (train/val/test)
            dataset_id: Unique identifier
            
        Returns:
            Dictionary with lock information
        """
        # Create hash for integrity verification
        data_json = data.to_json()
        data_hash = hashlib.sha256(data_json.encode()).hexdigest()
        
        # Encrypt data
        encrypted_data = self.cipher.encrypt(data_json.encode())
        
        # Save encrypted data
        encrypted_path = self.vault_path / f"{dataset_id}_{data_type.value}.enc"
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Record in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        max_access = 1 if data_type == DataType.TEST else 999999
        is_locked = 1 if data_type == DataType.TEST else 0
        
        cursor.execute('''
            INSERT INTO datasets 
            (dataset_id, data_type, creation_date, data_hash, encrypted_path, max_access, is_locked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id,
            data_type.value,
            datetime.now().isoformat(),
            data_hash,
            str(encrypted_path),
            max_access,
            is_locked
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(
            f"Dataset locked: {dataset_id} ({data_type.value}), "
            f"hash={data_hash[:16]}..."
        )
        
        return {
            'dataset_id': dataset_id,
            'data_type': data_type.value,
            'hash': data_hash,
            'is_locked': bool(is_locked),
            'max_access': max_access
        }
    
    def request_access(
        self,
        dataset_id: str,
        user: str,
        strategy_id: str,
        justification: str
    ) -> Optional[pd.DataFrame]:
        """
        Request access to a dataset
        
        For test sets, this enforces one-time access
        
        Args:
            dataset_id: Dataset to access
            user: User requesting access
            strategy_id: Strategy being validated
            justification: Reason for access
            
        Returns:
            DataFrame if access granted, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get dataset info
        cursor.execute('''
            SELECT data_type, access_count, max_access, is_locked, encrypted_path, data_hash
            FROM datasets
            WHERE dataset_id = ?
        ''', (dataset_id,))
        
        row = cursor.fetchone()
        
        if row is None:
            self._log_access(
                conn, dataset_id, user, strategy_id,
                granted=False, reason="Dataset not found"
            )
            conn.close()
            return None
        
        data_type, access_count, max_access, is_locked, encrypted_path, data_hash = row
        
        # Check if access allowed
        if access_count >= max_access:
            self._log_access(
                conn, dataset_id, user, strategy_id,
                granted=False, 
                reason=f"Max access ({max_access}) already reached. Data compromised."
            )
            
            self.logger.warning(
                f"DENIED: {user} attempted to access {dataset_id} "
                f"(already accessed {access_count} times)"
            )
            
            conn.close()
            return None
        
        # Grant access
        cursor.execute('''
            UPDATE datasets
            SET access_count = access_count + 1
            WHERE dataset_id = ?
        ''', (dataset_id,))
        
        self._log_access(
            conn, dataset_id, user, strategy_id,
            granted=True, reason=justification
        )
        
        conn.commit()
        conn.close()
        
        # Load and decrypt data
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        data_json = decrypted_data.decode()
        
        # Verify integrity
        current_hash = hashlib.sha256(data_json.encode()).hexdigest()
        if current_hash != data_hash:
            self.logger.error(
                f"INTEGRITY ERROR: Hash mismatch for {dataset_id}"
            )
            raise ValueError("Data integrity check failed!")
        
        # Parse data
        data = pd.read_json(data_json)
        
        self.logger.info(
            f"GRANTED: {user} accessed {dataset_id} (strategy={strategy_id})"
        )
        
        # If test set, send alert
        if data_type == 'test':
            self._send_critical_alert(
                f"TEST SET ACCESSED: {dataset_id} by {user} for {strategy_id}"
            )
        
        return data
    
    def _log_access(
        self,
        conn: sqlite3.Connection,
        dataset_id: str,
        user: str,
        strategy_id: str,
        granted: bool,
        reason: str
    ):
        """Log access attempt"""
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO access_log
            (timestamp, user, dataset_id, strategy_id, access_granted, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user,
            dataset_id,
            strategy_id,
            1 if granted else 0,
            reason
        ))
    
    def report_validation_results(
        self,
        dataset_id: str,
        user: str,
        strategy_id: str,
        performance_metrics: Dict,
        decision: str
    ):
        """
        Report validation results (mandatory after test set access)
        
        Args:
            dataset_id: Dataset used
            user: User who ran validation
            strategy_id: Strategy validated
            performance_metrics: Performance metrics
            decision: Final decision (approve/reject/conditional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store results
        cursor.execute('''
            INSERT INTO validation_results
            (timestamp, user, strategy_id, dataset_id, performance_metrics, decision)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user,
            strategy_id,
            dataset_id,
            json.dumps(performance_metrics),
            decision
        ))
        
        # Mark as reported in access log
        cursor.execute('''
            UPDATE access_log
            SET results_reported = 1
            WHERE dataset_id = ? AND user = ? AND strategy_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (dataset_id, user, strategy_id))
        
        conn.commit()
        conn.close()
        
        self.logger.info(
            f"Results reported: {strategy_id} on {dataset_id} - {decision}"
        )
    
    def get_unreported_validations(self) -> List[Dict]:
        """
        Get list of test set accesses without reported results
        
        Used for compliance monitoring
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT al.timestamp, al.user, al.strategy_id, al.dataset_id
            FROM access_log al
            JOIN datasets d ON al.dataset_id = d.dataset_id
            WHERE d.data_type = 'test' 
            AND al.access_granted = 1
            AND al.results_reported = 0
        ''')
        
        unreported = []
        for row in cursor.fetchall():
            unreported.append({
                'timestamp': row[0],
                'user': row[1],
                'strategy_id': row[2],
                'dataset_id': row[3]
            })
        
        conn.close()
        return unreported
    
    def generate_audit_report(self) -> pd.DataFrame:
        """Generate comprehensive audit report"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                al.timestamp,
                al.user,
                al.strategy_id,
                al.dataset_id,
                d.data_type,
                al.access_granted,
                al.results_reported,
                vr.decision
            FROM access_log al
            LEFT JOIN datasets d ON al.dataset_id = d.dataset_id
            LEFT JOIN validation_results vr 
                ON al.strategy_id = vr.strategy_id 
                AND al.dataset_id = vr.dataset_id
            ORDER BY al.timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def _send_critical_alert(self, message: str):
        """Send critical alert (email, Slack, PagerDuty)"""
        # In production, integrate with alerting system
        self.logger.critical(f"ALERT: {message}")
        print(f"\\n🚨 CRITICAL ALERT: {message}\\n")


# Example usage and workflow
def example_secure_workflow():
    """Demonstrate secure test set workflow"""
    
    # Initialize vault
    vault = SecureTestSetVault(
        vault_path="/secure/vault",
        db_path="/secure/access_control.db"
    )
    
    # Step 1: Lock datasets (done once at start)
    np.random.seed(42)
    
    train_data = pd.DataFrame({
        'date': pd.date_range('2018-01-01', periods=1000, freq='D'),
        'returns': np.random.randn(1000) * 0.01
    })
    
    val_data = pd.DataFrame({
        'date': pd.date_range('2020-09-01', periods=300, freq='D'),
        'returns': np.random.randn(300) * 0.01
    })
    
    test_data = pd.DataFrame({
        'date': pd.date_range('2021-07-01', periods=300, freq='D'),
        'returns': np.random.randn(300) * 0.01
    })
    
    # Lock all datasets
    vault.lock_dataset(train_data, DataType.TRAINING, "dataset_v1")
    vault.lock_dataset(val_data, DataType.VALIDATION, "dataset_v1")
    vault.lock_dataset(test_data, DataType.TEST, "dataset_v1")
    
    print("\\n" + "="*80)
    print("SECURE WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Step 2: Researcher accesses train/val (allowed multiple times)
    print("\\n1. Researcher developing strategy...")
    train = vault.request_access(
        "dataset_v1_training",
        user="alice@fund.com",
        strategy_id="momentum_v1",
        justification="Strategy development"
    )
    print("✓ Training data accessed")
    
    validation = vault.request_access(
        "dataset_v1_validation",
        user="alice@fund.com",
        strategy_id="momentum_v1",
        justification="Parameter validation"
    )
    print("✓ Validation data accessed")
    
    # Step 3: Request test set access (ONLY ONCE)
    print("\\n2. Final validation on test set...")
    test = vault.request_access(
        "dataset_v1_test",
        user="alice@fund.com",
        strategy_id="momentum_v1",
        justification="Final out-of-sample validation"
    )
    
    if test is not None:
        print("✓ Test data accessed (1st time - ALLOWED)")
        
        # Step 4: MUST report results
        vault.report_validation_results(
            dataset_id="dataset_v1_test",
            user="alice@fund.com",
            strategy_id="momentum_v1",
            performance_metrics={'sharpe': 1.2, 'max_dd': -0.15},
            decision="APPROVED"
        )
        print("✓ Results reported")
    
    # Step 5: Try to access test set again (DENIED)
    print("\\n3. Attempting second test set access...")
    test2 = vault.request_access(
        "dataset_v1_test",
        user="alice@fund.com",
        strategy_id="momentum_v2",  # Different strategy!
        justification="Testing modified strategy"
    )
    
    if test2 is None:
        print("✗ Access DENIED - test set already used")
    
    # Step 6: Audit report
    print("\\n4. Generating audit report...")
    audit = vault.generate_audit_report()
    print(audit)
    
    # Step 7: Check for unreported validations
    unreported = vault.get_unreported_validations()
    if unreported:
        print(f"\\n⚠️  {len(unreported)} unreported validations!")
        for item in unreported:
            print(f"  - {item['user']}: {item['strategy_id']}")


if __name__ == "__main__":
    example_secure_workflow()
\`\`\`

**Key System Features:**

1. **Cryptographic Encryption**: Test data encrypted at rest
2. **One-Time Access**: Database-enforced single access to test sets
3. **Audit Logging**: Every access attempt logged with timestamp, user, justification
4. **Integrity Verification**: SHA-256 hashes ensure data hasn't been tampered with
5. **Mandatory Reporting**: System tracks whether results were reported
6. **Critical Alerts**: Real-time notifications when test sets are accessed
7. **Compliance Monitoring**: Automated detection of unreported validations

**Organizational Processes:**

1. **Pre-registration**: Strategies must be registered before test access
2. **Investment Committee Review**: Test results reviewed before deployment
3. **Penalties**: Researchers who violate test set policy face consequences
4. **Training**: Mandatory training on proper OOS testing procedures
5. **Quarterly Audits**: Review all test set accesses and results

This system makes it technologically impossible to accidentally or intentionally compromise test data while maintaining research velocity.

---

## Question 2: Handling Failed Out-of-Sample Tests

**Scenario**: Your team has developed a sophisticated machine learning strategy that shows exceptional performance:
- Training Set (3 years): Sharpe 2.8
- Validation Set (6 months): Sharpe 2.3
- Test Set (6 months): Sharpe 0.4

The strategy passes all other checks (no look-ahead bias, proper transaction costs, statistically significant on train/val). But the OOS test shows severe degradation. The lead researcher argues: "The test period was unusual—COVID volatility. We should test on a different period."

**Questions:**
1. How should you handle this failed OOS test?
2. Is the researcher's argument valid?
3. What options exist going forward?
4. How do you prevent similar situations?

### Comprehensive Answer

This is a moment of truth that tests organizational discipline. The correct answer is uncomfortable but clear: **the test has failed, and the strategy should not be deployed in its current form.**

**Analysis of the Situation:**

The 86% degradation (from 2.8 to 0.4 Sharpe) is catastrophic. Even accounting for an "unusual" test period, a robust strategy shouldn't collapse this severely. The performance pattern suggests classic overfitting:
- High training performance
- Moderate validation performance  
- Terrible test performance

This is exactly the curve-fitting signature machine learning models create when they memorize training patterns without learning general principles.

**The Researcher's Argument:**

The argument "test period was unusual, let's try another" is **INVALID** and represents a fundamental misunderstanding of scientific method. Here's why:

\`\`\`python
class FailedOOSHandler:
    """
    Framework for handling failed out-of-sample tests
    """
    
    def evaluate_retest_request(
        self,
        original_test_sharpe: float,
        train_sharpe: float,
        val_sharpe: float,
        researcher_justification: str
    ) -> Dict:
        """
        Evaluate whether a retest request is valid
        
        Args:
            original_test_sharpe: Sharpe on original test set
            train_sharpe: Training Sharpe
            val_sharpe: Validation Sharpe
            researcher_justification: Researcher's reason for retest
            
        Returns:
            Evaluation decision
        """
        degradation = (train_sharpe - original_test_sharpe) / train_sharpe * 100
        
        # Red flags
        red_flags = []
        
        if degradation > 50:
            red_flags.append("Severe degradation (>50%) indicates overfitting")
        
        if abs(val_sharpe - original_test_sharpe) / val_sharpe * 100 > 50:
            red_flags.append("Large gap between validation and test suggests validation contamination")
        
        # Common invalid justifications
        invalid_patterns = [
            "test period was unusual",
            "different time period",
            "market conditions changed",
            "COVID", "pandemic", "crisis"  # Using crisis as excuse
        ]
        
        if any(pattern.lower() in researcher_justification.lower() 
               for pattern in invalid_patterns):
            red_flags.append(
                "Justification attempts to invalidate test period - "
                "this defeats the purpose of OOS testing"
            )
        
        decision = {
            'allow_retest': False,
            'red_flags': red_flags,
            'recommendation': self._generate_recommendation(degradation, red_flags)
        }
        
        return decision
    
    def _generate_recommendation(
        self,
        degradation: float,
        red_flags: List[str]
    ) -> str:
        """Generate recommendation for failed OOS test"""
        
        if degradation > 70:
            return (
                "REJECT: Severe overfitting detected. Strategy memorized training patterns "
                "that don't generalize. Options: "
                "(1) Abandon strategy, "
                "(2) Complete redesign with focus on simplicity/robustness, "
                "(3) Revisit in 1+ years with completely new data for honest retest."
            )
        elif degradation > 50:
            return (
                "CONDITIONAL REJECT: Significant degradation. Options: "
                "(1) Extensive paper trading (6+ months) with minimal capital, "
                "(2) Redesign to increase robustness, "
                "(3) If paper trading succeeds, cautious deployment with tight risk limits."
            )
        else:
            return (
                "MARGINAL: Performance degradation concerning but not catastrophic. "
                "Extended paper trading recommended."
            )


# Demonstrate the scenario
handler = FailedOOSHandler()

evaluation = handler.evaluate_retest_request(
    original_test_sharpe=0.4,
    train_sharpe=2.8,
    val_sharpe=2.3,
    researcher_justification="The test period included COVID volatility which was unusual"
)

print("\\nFAILED OOS TEST EVALUATION")
print("="*80)
print(f"Allow Retest: {evaluation['allow_retest']}")
print(f"\\nRed Flags:")
for flag in evaluation['red_flags']:
    print(f"  🚩 {flag}")
print(f"\\nRecommendation:\\n{evaluation['recommendation']}")
print("="*80)
\`\`\`

**Why "Unusual Period" is Not Valid:**

1. **Defeats Purpose**: OOS testing exists precisely to test robustness under various conditions, including unusual ones
2. **Cherry-Picking**: Choosing a "better" test period is just another form of overfitting
3. **Real World**: Live trading encounters unusual periods all the time
4. **Slippery Slope**: Every failed test can be excused as "unusual"

**The Correct Response:**

"The test period WAS unusual—that's exactly why the strategy failed. A robust strategy should handle unusual periods reasonably well, or at minimum not collapse completely. The 86% degradation indicates fundamental overfitting, not bad luck."

**Valid Options Going Forward:**

**Option 1: Abandon Strategy (Recommended)**
- 86% degradation is too severe
- ML model likely overfit to training regime
- Cost-benefit doesn't justify further investment

**Option 2: Major Redesign**
- Simplify model dramatically (reduce parameters by 50%+)
- Add explicit regime detection
- Focus on robustness over performance
- Start fresh with new test set (1+ year from now)

**Option 3: Cautious Paper Trading**
- Deploy in paper trading only
- Zero real capital
- Monitor for 6+ months
- If paper trading succeeds, minimal capital deployment ($100K vs typical $5M+)
- Strict stop-loss (halt if drawdown >10%)

**Option 4: Honest Retest (in 1+ years)**
- Wait for completely new data accumulation
- No changes to strategy in meantime
- One final honest test on truly unseen data
- If this fails too, strategy is dead

**The Option That's NOT Valid:**

❌ "Let's just retest on 2019 data" - This is p-hacking with test sets

**Prevention for Future:**

1. **Pre-register strategies** with parameters and test dates before any testing
2. **Require economic rationale** not just statistical patterns
3. **Ensemble methods** often more robust than single models
4. **Simpler models** (regularization, feature selection)
5. **Multiple test sets** locked away at different times
6. **Independent validation team** not involved in development

**The Hard Truth:**

Most sophisticated ML strategies fail OOS tests. Success rate in professional quant funds is roughly:
- 30% of strategies pass validation
- 20% of those survive first year live
- Only ~6% become long-term profitable

Failed OOS tests are the norm, not the exception. The key is accepting them gracefully and learning, not rationalizing them away.

---

## Question 3: Test Set Size and Power Analysis

**Scenario**: You're establishing policies for test set requirements. Different strategy types have different characteristics:

- **HFT**: Thousands of trades per day, very low per-trade profits
- **Daily mean reversion**: 5-20 trades per day
- **Swing trading**: 2-5 trades per week  
- **Position trading**: 1-2 trades per month

**Questions:**
1. What minimum test set size (in time and number of trades) should each strategy type have?
2. How do you handle strategies with very few trades in the test period?
3. Should test set requirements be flexible or fixed?

### Comprehensive Answer

Test set requirements must balance statistical power with practical constraints, varying by strategy frequency and holding period.

**Framework for Test Set Sizing:**

\`\`\`python
def calculate_required_test_size(
    strategy_frequency: str,
    target_sharpe: float,
    desired_power: float = 0.80,
    significance: float = 0.05
) -> Dict:
    """
    Calculate required test set size for adequate statistical power
    
    Args:
        strategy_frequency: 'hft', 'daily', 'swing', 'position'
        target_sharpe: Expected Sharpe ratio
        desired_power: Desired statistical power (typically 0.80)
        significance: Significance level (typically 0.05)
        
    Returns:
        Required sample sizes
    """
    from scipy import stats
    
    # Calculate required n for given power
    # Using formula: n = ((z_alpha + z_beta) / effect_size)^2
    # Effect size for Sharpe ratio test
    
    z_alpha = stats.norm.ppf(1 - significance / 2)  # Two-tailed
    z_beta = stats.norm.ppf(desired_power)
    
    # Approximate required observations
    # This is simplified; actual calculation more complex
    required_n = int(((z_alpha + z_beta) / (target_sharpe / np.sqrt(1 + target_sharpe**2 / 2)))**2)
    
    # Translate to time periods by frequency
    frequency_map = {
        'hft': {
            'trades_per_day': 1000,
            'min_days': 20,  # Even HFT needs multiple days
            'min_trades': 10000
        },
        'daily': {
            'trades_per_day': 10,
            'min_days': max(required_n / 10, 60),  # At least 60 days
            'min_trades': max(required_n, 200)
        },
        'swing': {
            'trades_per_day': 0.5,
            'min_days': max(required_n / 0.5, 120),  # At least 120 days
            'min_trades': max(required_n, 50)
        },
        'position': {
            'trades_per_day': 0.1,
            'min_days': max(required_n / 0.1, 240),  # At least 240 days
            'min_trades': max(required_n, 20)
        }
    }
    
    freq_data = frequency_map[strategy_frequency.lower()]
    
    return {
        'required_observations': required_n,
        'min_calendar_days': int(freq_data['min_days']),
        'min_trades': int(freq_data['min_trades']),
        'expected_trades_in_period': int(freq_data['min_days'] * freq_data['trades_per_day']),
        'power': desired_power,
        'interpretation': _interpret_requirements(strategy_frequency, freq_data)
    }

def _interpret_requirements(freq: str, data: Dict) -> str:
    """Interpret test requirements"""
    days = data['min_days']
    trades = data['min_trades']
    months = days / 30
    
    return (
        f"For {freq} strategy: Minimum {days} calendar days (~{months:.1f} months) "
        f"with at least {trades} executed trades for adequate statistical power."
    )

# Calculate for each strategy type
print("\\nTEST SET SIZE REQUIREMENTS")
print("="*80)

strategies = {
    'HFT': ('hft', 1.5),
    'Daily Mean Reversion': ('daily', 1.2),
    'Swing Trading': ('swing', 1.0),
    'Position Trading': ('position', 0.8)
}

for name, (freq, sharpe) in strategies.items():
    requirements = calculate_required_test_size(freq, sharpe)
    
    print(f"\\n{name} (Expected Sharpe: {sharpe}):")
    print(f"  Min calendar days: {requirements['min_calendar_days']}")
    print(f"  Min trades: {requirements['min_trades']}")
    print(f"  Expected trades: {requirements['expected_trades_in_period']}")
    print(f"  {requirements['interpretation']}")
\`\`\`

**Recommended Minimums:**

| Strategy Type | Min Time Period | Min Trades | Rationale |
|--------------|----------------|------------|-----------|
| HFT | 20 days | 10,000 | High frequency needs less time but many events |
| Daily | 60 days (3 months) | 200 | Standard for daily strategies |
| Swing | 120 days (6 months) | 50 | Lower frequency needs longer period |
| Position | 240 days (1 year) | 20 | Very low frequency needs extended time |

**Handling Low-Trade Strategies:**

For strategies with very few trades (position trading):
- Cannot rely on statistical significance alone
- Must emphasize:
  - Economic rationale
  - Comparison to similar strategies
  - Extended paper trading
  - Lower capital allocation
  - Tighter monitoring

**Policy Recommendation:**

**Fixed Minimums with Conditional Approval:**
- Set absolute minimums (time AND trades)
- Below minimums = automatic conditional status
- Conditional strategies:
  - Extended paper trading required
  - Reduced capital allocation (50% of typical)
  - More frequent review (monthly vs quarterly)
  - Investment committee approval for each trade

**Example Policy:**

\`\`\`
TEST SET REQUIREMENTS - FIRM POLICY

Tier 1 (Standard Approval):
- Daily strategies: >= 6 months AND >= 200 trades
- Lower frequency: >= 1 year AND >= 50 trades
- Test set Sharpe >= 1.0
- Degradation < 30%

Tier 2 (Conditional Approval):
- Meets time OR trades requirement (not both)
- Test set Sharpe >= 0.5
- 6+ months paper trading required
- Max allocation: $1M (vs $5M typical)

Tier 3 (Reject):
- Fails to meet minimums
- Test set Sharpe < 0.5
- Degradation > 50%
- No economic rationale
\`\`\`

This balances statistical rigor with business practicality while maintaining scientific standards.
`,
    },
  ],
};

export default outOfSampleTestingDiscussion;
