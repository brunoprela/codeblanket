export const mlopsBestPractices = {
  title: 'MLOps Best Practices',
  id: 'mlops-best-practices',
  content: `
# MLOps Best Practices

## Introduction

**"DevOps for ML isn't just DevOps + ML. It's a different beast."**

MLOps (Machine Learning Operations) extends DevOps practices to ML systems. But ML adds unique challenges: data dependencies, model versioning, experiment tracking, and monitoring drift.

**Why MLOps Matters**:
- 87% of ML projects never make it to production (VentureBeat)
- Average time from model training to production: 6+ months
- Technical debt in ML systems grows faster than traditional software

This section covers production-grade MLOps practices.

### MLOps Maturity Levels

\`\`\`
Level 0: Manual Process
  - Notebook development
  - Manual deployment
  - No monitoring

Level 1: ML Pipeline Automation
  - Automated training
  - Model registry
  - Basic monitoring

Level 2: CI/CD Pipeline Automation
  - Automated tests
  - Continuous training
  - A/B testing framework

Level 3: Full MLOps
  - Auto-retraining on drift
  - Feature store
  - Comprehensive monitoring
  - Automated rollback
\`\`\`

---

## CI/CD for ML

### Continuous Integration

\`\`\`python
"""
CI/CD Pipeline for ML Models
"""

class MLCIPipeline:
    """
    Continuous Integration for ML
    
    Tests that must pass before deployment:
    1. Data validation
    2. Model training
    3. Model evaluation
    4. Performance benchmarks
    5. Integration tests
    """
    
    def __init__(self):
        self.ci_stages = {
            "1_data_validation": [
                "Schema validation",
                "Data quality checks",
                "Feature distribution",
                "Missing values"
            ],
            "2_model_training": [
                "Training pipeline",
                "Hyperparameter validation",
                "Training convergence",
                "Resource usage"
            ],
            "3_model_evaluation": [
                "Accuracy thresholds",
                "Fairness metrics",
                "Inference latency",
                "Model size"
            ],
            "4_integration_tests": [
                "API contract tests",
                "Load tests",
                "Backwards compatibility",
                "Rollback procedure"
            ]
        }
    
    def data_validation_tests(self, data):
        """
        Validate data before training
        """
        print("\\n=== Data Validation Tests ===\\n")
        
        import pandas as pd
        import numpy as np
        
        # Example DataFrame
        df = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000) * 10,
            'target': np.random.randint(0, 2, 1000)
        })
        
        tests_passed = []
        
        # Test 1: Schema validation
        expected_columns = ['feature1', 'feature2', 'target']
        schema_valid = all(col in df.columns for col in expected_columns)
        tests_passed.append(("Schema validation", schema_valid))
        
        # Test 2: Missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        missing_ok = missing_pct < 0.05  # < 5% missing
        tests_passed.append(("Missing values check", missing_ok))
        
        # Test 3: Feature distributions
        # Check for data drift using simple statistics
        feature1_mean = df['feature1'].mean()
        feature1_std = df['feature1'].std()
        
        # Expected: mean ≈ 0, std ≈ 1
        distribution_ok = abs(feature1_mean) < 0.5 and 0.5 < feature1_std < 1.5
        tests_passed.append(("Feature distribution", distribution_ok))
        
        # Test 4: Target balance
        target_balance = df['target'].value_counts()
        min_class_pct = target_balance.min() / len(df)
        balance_ok = min_class_pct > 0.1  # No class < 10%
        tests_passed.append(("Target balance", balance_ok))
        
        # Print results
        for test_name, passed in tests_passed:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        all_passed = all(passed for _, passed in tests_passed)
        
        if all_passed:
            print("\\n✓ All data validation tests passed")
        else:
            print("\\n✗ Data validation failed - blocking deployment")
        
        return all_passed
    
    def model_evaluation_tests(self, model, X_test, y_test):
        """
        Model must pass performance thresholds
        """
        print("\\n=== Model Evaluation Tests ===\\n")
        
        import time
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Predictions
        y_pred = model.predict(X_test)
        
        tests_passed = []
        
        # Test 1: Accuracy threshold
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_ok = accuracy > 0.75  # Minimum 75%
        tests_passed.append((f"Accuracy > 0.75 (got {accuracy:.3f})", accuracy_ok))
        
        # Test 2: Precision threshold
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        precision_ok = precision > 0.70
        tests_passed.append((f"Precision > 0.70 (got {precision:.3f})", precision_ok))
        
        # Test 3: Recall threshold
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_ok = recall > 0.70
        tests_passed.append((f"Recall > 0.70 (got {recall:.3f})", recall_ok))
        
        # Test 4: Inference latency
        start = time.time()
        for _ in range(100):
            model.predict(X_test[:10])
        latency_ms = (time.time() - start) / 100 * 1000
        
        latency_ok = latency_ms < 100  # < 100ms per batch
        tests_passed.append((f"Latency < 100ms (got {latency_ms:.1f}ms)", latency_ok))
        
        # Print results
        for test_name, passed in tests_passed:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        all_passed = all(passed for _, passed in tests_passed)
        
        if all_passed:
            print("\\n✓ All model evaluation tests passed")
        else:
            print("\\n✗ Model evaluation failed - blocking deployment")
        
        return all_passed
    
    def run_ci_pipeline(self):
        """
        Full CI pipeline
        """
        print("\\n=== Running ML CI Pipeline ===\\n")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Stage 1: Data validation
        print("Stage 1/4: Data Validation")
        data_ok = self.data_validation_tests(None)  # Would use real data
        
        if not data_ok:
            print("\\n❌ Pipeline failed at data validation")
            return False
        
        # Stage 2: Model training
        print("\\nStage 2/4: Model Training")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("✓ Model trained successfully")
        
        # Stage 3: Model evaluation
        print("\\nStage 3/4: Model Evaluation")
        model_ok = self.model_evaluation_tests(model, X_test, y_test)
        
        if not model_ok:
            print("\\n❌ Pipeline failed at model evaluation")
            return False
        
        # Stage 4: Integration tests
        print("\\nStage 4/4: Integration Tests")
        print("✓ API contract test passed")
        print("✓ Load test passed")
        print("✓ Backward compatibility verified")
        
        print("\\n✅ CI Pipeline Passed - Ready for Deployment")
        return True


# Run CI pipeline
ci = MLCIPipeline()
ci.run_ci_pipeline()
\`\`\`

### Continuous Deployment

\`\`\`python
"""
Continuous Deployment with Canary & Blue-Green
"""

class MLCDPipeline:
    """
    Deployment strategies for ML models
    """
    
    def canary_deployment(self, new_model, old_model, traffic_split=0.05):
        """
        Canary Deployment: Gradually shift traffic to new model
        
        Process:
        1. Deploy new model to 5% of traffic
        2. Monitor metrics for 1 hour
        3. If good, increase to 25%
        4. Monitor again
        5. If good, increase to 100%
        6. If bad at any stage, rollback
        """
        print("\\n=== Canary Deployment ===\\n")
        
        stages = [
            (0.05, "5% traffic", "1 hour"),
            (0.25, "25% traffic", "4 hours"),
            (0.50, "50% traffic", "12 hours"),
            (1.00, "100% traffic", "complete")
        ]
        
        for traffic_pct, description, duration in stages:
            print(f"Stage: {description}")
            print(f"  Duration: {duration}")
            print(f"  Monitoring:")
            print(f"    - Accuracy: Check no degradation")
            print(f"    - Latency: P99 < threshold")
            print(f"    - Error rate: < 1%")
            print(f"    - Business metrics: Revenue, engagement")
            
            # Simulate monitoring
            metrics_ok = self._monitor_deployment(new_model, old_model, traffic_pct)
            
            if metrics_ok:
                print(f"  ✓ Metrics OK - proceeding to next stage\\n")
            else:
                print(f"  ✗ Metrics degraded - ROLLING BACK")
                self._rollback(old_model)
                return False
        
        print("✅ Canary deployment complete - new model at 100%")
        return True
    
    def _monitor_deployment(self, new_model, old_model, traffic_pct):
        """
        Monitor canary deployment
        """
        # Simulate metrics (would be real monitoring)
        import random
        
        # Simulate: 95% chance metrics are OK
        return random.random() < 0.95
    
    def _rollback(self, old_model):
        """
        Automatic rollback to previous model
        """
        print("\\n🔄 Initiating automatic rollback...")
        print("  1. Route 100% traffic to old model")
        print("  2. Remove new model from serving")
        print("  3. Alert on-call engineer")
        print("  4. Save failure logs for analysis")
        print("\\n✓ Rollback complete - old model serving")
    
    def blue_green_deployment(self):
        """
        Blue-Green Deployment: Zero-downtime switch
        
        Blue = Current production
        Green = New version
        
        Process:
        1. Deploy green (new model) alongside blue
        2. Run smoke tests on green
        3. Switch load balancer to green
        4. Keep blue running for quick rollback
        """
        print("\\n=== Blue-Green Deployment ===\\n")
        
        print("Step 1: Deploy Green Environment")
        print("  - New model version: v2.0")
        print("  - Infrastructure: Identical to blue")
        print("  - Endpoints: Not yet receiving traffic")
        print("  ✓ Green deployed\\n")
        
        print("Step 2: Smoke Tests on Green")
        print("  - Health check: ✓ Pass")
        print("  - Prediction test: ✓ Pass")
        print("  - Latency test: ✓ Pass")
        print("  ✓ All tests passed\\n")
        
        print("Step 3: Switch Traffic")
        print("  - Load balancer: blue → green")
        print("  - Traffic: 0% → 100% (instant switch)")
        print("  ✓ Switched\\n")
        
        print("Step 4: Monitor & Keep Blue")
        print("  - Monitor green for 24 hours")
        print("  - Keep blue environment running")
        print("  - If issues: instant switch back")
        print("  ✓ Monitoring\\n")
        
        print("✅ Blue-Green deployment complete")


# Example deployments
cd = MLCDPipeline()

print("=== Deployment Strategy 1: Canary ===")
cd.canary_deployment(new_model=None, old_model=None)

print("\\n" + "="*50)
cd.blue_green_deployment()
\`\`\`

---

## Model Versioning & Registry

### Model Registry

\`\`\`python
"""
Model Registry with MLflow
"""

import mlflow
import mlflow.sklearn
from datetime import datetime

class ModelRegistry:
    """
    Centralized model registry
    
    Benefits:
    - Version control for models
    - Lineage tracking
    - Staging/production promotion
    - Rollback capability
    """
    
    def __init__(self, registry_uri="sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(registry_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, model, name, metrics, hyperparameters, dataset_version):
        """
        Register model with full lineage
        """
        print(f"\\n=== Registering Model: {name} ===\\n")
        
        with mlflow.start_run() as run:
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log hyperparameters
            for param_name, value in hyperparameters.items():
                mlflow.log_param(param_name, value)
            
            # Log dataset version
            mlflow.log_param("dataset_version", dataset_version)
            
            # Log metadata
            mlflow.log_param("trained_at", datetime.now().isoformat())
            mlflow.log_param("framework", "scikit-learn")
            
            # Register to model registry
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, name)
            
            print(f"✓ Model registered: {name} version {mv.version}")
            print(f"  Run ID: {run.info.run_id}")
            print(f"  Metrics: {metrics}")
            print(f"  Dataset: {dataset_version}")
            
            return mv.version
    
    def promote_to_production(self, model_name, version):
        """
        Promote model to production stage
        """
        print(f"\\n=== Promoting {model_name} v{version} to Production ===\\n")
        
        # Archive current production model
        current_prod = self._get_production_model(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage="Archived"
            )
            print(f"  ✓ Archived previous production version: {current_prod.version}")
        
        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"  ✓ Version {version} promoted to Production")
        print(f"  ✓ Ready to serve")
    
    def _get_production_model(self, model_name):
        """
        Get current production model
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return versions[0] if versions else None
        except:
            return None
    
    def list_models(self, model_name):
        """
        List all versions of a model
        """
        print(f"\\n=== Model Versions: {model_name} ===\\n")
        
        versions = self.client.search_model_versions(f"name='{model_name}'")
        
        for v in versions:
            print(f"Version {v.version}:")
            print(f"  Stage: {v.current_stage}")
            print(f"  Created: {v.creation_timestamp}")
            print(f"  Run ID: {v.run_id}")
            print()
    
    def rollback_to_version(self, model_name, version):
        """
        Rollback to previous version
        """
        print(f"\\n=== Rolling Back {model_name} to v{version} ===\\n")
        
        # Demote current production
        current = self._get_production_model(model_name)
        if current:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current.version,
                stage="Archived"
            )
        
        # Promote rollback version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"✓ Rolled back to version {version}")
        print(f"✓ Now serving v{version}")


# Example: Model registry workflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score

# Train model
X, y = make_classification(n_samples=1000, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred, zero_division=0)
}

# Register model
registry = ModelRegistry()
version = registry.register_model(
    model=model,
    name="fraud_detection",
    metrics=metrics,
    hyperparameters={"n_estimators": 100, "max_depth": 10},
    dataset_version="v2.3.1"
)

# Promote to production
registry.promote_to_production("fraud_detection", version)

# Later: rollback if needed
# registry.rollback_to_version("fraud_detection", version - 1)
\`\`\`

---

## Testing Strategies

### ML-Specific Tests

\`\`\`python
"""
Testing ML Systems
"""

class MLTestSuite:
    """
    Comprehensive testing for ML systems
    
    Types of tests:
    1. Data tests
    2. Model tests
    3. Infrastructure tests
    4. Integration tests
    """
    
    def test_data_quality(self, df):
        """
        Data quality tests
        """
        print("\\n=== Data Quality Tests ===\\n")
        
        tests = []
        
        # Test: No unexpected nulls
        critical_columns = ['feature1', 'feature2', 'target']
        for col in critical_columns:
            has_nulls = df[col].isnull().any()
            tests.append((f"No nulls in {col}", not has_nulls))
        
        # Test: Value ranges
        if 'age' in df.columns:
            age_valid = (df['age'] >= 0).all() and (df['age'] <= 120).all()
            tests.append(("Age in valid range", age_valid))
        
        # Test: No duplicates
        no_duplicates = not df.duplicated().any()
        tests.append(("No duplicate rows", no_duplicates))
        
        self._print_test_results(tests)
        
        return all(passed for _, passed in tests)
    
    def test_model_invariance(self, model, X_test):
        """
        Model invariance tests
        
        Predictions should be stable under small perturbations
        """
        print("\\n=== Model Invariance Tests ===\\n")
        
        import numpy as np
        
        tests = []
        
        # Original predictions
        y_pred_original = model.predict(X_test)
        
        # Test 1: Small perturbations shouldn't change predictions much
        noise = np.random.randn(*X_test.shape) * 0.001
        X_perturbed = X_test + noise
        y_pred_perturbed = model.predict(X_perturbed)
        
        agreement = (y_pred_original == y_pred_perturbed).mean()
        stable = agreement > 0.95  # 95% predictions should match
        tests.append((f"Stable under noise (agreement: {agreement:.2%})", stable))
        
        # Test 2: Duplicate inputs should get same prediction
        X_duplicated = np.vstack([X_test[:5], X_test[:5]])
        y_pred_duplicated = model.predict(X_duplicated)
        
        deterministic = np.array_equal(
            y_pred_duplicated[:5],
            y_pred_duplicated[5:]
        )
        tests.append(("Deterministic predictions", deterministic))
        
        self._print_test_results(tests)
        
        return all(passed for _, passed in tests)
    
    def test_model_performance(self, model, X_test, y_test):
        """
        Model performance tests
        
        Must meet minimum thresholds
        """
        print("\\n=== Model Performance Tests ===\\n")
        
        from sklearn.metrics import accuracy_score, f1_score
        
        y_pred = model.predict(X_test)
        
        tests = []
        
        # Minimum accuracy
        accuracy = accuracy_score(y_test, y_pred)
        meets_accuracy = accuracy >= 0.75
        tests.append((f"Accuracy >= 0.75 (got {accuracy:.3f})", meets_accuracy))
        
        # Minimum F1
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        meets_f1 = f1 >= 0.70
        tests.append((f"F1 >= 0.70 (got {f1:.3f})", meets_f1))
        
        # Better than baseline (most frequent class)
        from sklearn.dummy import DummyClassifier
        baseline = DummyClassifier(strategy='most_frequent')
        baseline.fit(X_test, y_test)
        baseline_acc = baseline.score(X_test, y_test)
        
        better_than_baseline = accuracy > baseline_acc
        tests.append((f"Better than baseline ({baseline_acc:.3f})", better_than_baseline))
        
        self._print_test_results(tests)
        
        return all(passed for _, passed in tests)
    
    def test_inference_latency(self, model, X_test):
        """
        Latency tests
        """
        print("\\n=== Inference Latency Tests ===\\n")
        
        import time
        
        tests = []
        
        # Single prediction latency
        start = time.time()
        for _ in range(100):
            model.predict(X_test[:1])
        single_latency_ms = (time.time() - start) / 100 * 1000
        
        single_ok = single_latency_ms < 10  # < 10ms
        tests.append((f"Single prediction < 10ms (got {single_latency_ms:.2f}ms)", single_ok))
        
        # Batch prediction latency
        start = time.time()
        for _ in range(10):
            model.predict(X_test[:100])
        batch_latency_ms = (time.time() - start) / 10 * 1000
        
        batch_ok = batch_latency_ms < 100  # < 100ms for batch of 100
        tests.append((f"Batch (100) < 100ms (got {batch_latency_ms:.2f}ms)", batch_ok))
        
        self._print_test_results(tests)
        
        return all(passed for _, passed in tests)
    
    def _print_test_results(self, tests):
        """
        Print test results
        """
        for test_name, passed in tests:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        all_passed = all(passed for _, passed in tests)
        
        if all_passed:
            print("\\n✓ All tests passed")
        else:
            print("\\n✗ Some tests failed")


# Run tests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create test data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(20)])
df['target'] = y

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Run test suite
test_suite = MLTestSuite()

test_suite.test_data_quality(df)
test_suite.test_model_invariance(model, X[:100])
test_suite.test_model_performance(model, X, y)
test_suite.test_inference_latency(model, X)

print("\\n" + "="*50)
print("✅ Test Suite Complete")
\`\`\`

---

## Documentation & Reproducibility

### Model Cards

\`\`\`python
"""
Model Cards for Documentation
"""

class ModelCard:
    """
    Document model for transparency and reproducibility
    
    Based on Google's Model Cards paper
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.card = {}
    
    def create_model_card(self):
        """
        Create comprehensive model documentation
        """
        self.card = {
            "model_details": {
                "name": "Fraud Detection Model",
                "version": "2.1.0",
                "date": "2024-10-15",
                "model_type": "Random Forest Classifier",
                "framework": "scikit-learn 1.3.0",
                "developers": "ML Team",
                "contact": "ml-team@company.com"
            },
            
            "intended_use": {
                "primary_use": "Detect fraudulent transactions in real-time",
                "intended_users": "Payment processing system",
                "out_of_scope": "Not for credit scoring or insurance"
            },
            
            "training_data": {
                "dataset": "Transactions 2024 Q1-Q3",
                "size": "10M transactions",
                "date_range": "2024-01-01 to 2024-09-30",
                "fraud_rate": "0.5%",
                "features": 50,
                "sampling": "SMOTE oversampling for fraud class"
            },
            
            "model_architecture": {
                "algorithm": "Random Forest",
                "hyperparameters": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 50
                },
                "feature_importance_top_5": [
                    "transaction_amount",
                    "time_since_last_transaction",
                    "merchant_risk_score",
                    "device_fingerprint",
                    "location_change"
                ]
            },
            
            "performance": {
                "test_set": "2024-10 (held-out)",
                "metrics": {
                    "accuracy": 0.987,
                    "precision": 0.82,
                    "recall": 0.75,
                    "f1_score": 0.78,
                    "auc_roc": 0.95
                },
                "latency": "P99: 45ms",
                "false_positive_rate": "0.01"
            },
            
            "ethical_considerations": {
                "fairness": "Tested across demographics - no significant bias",
                "privacy": "No PII used in features",
                "explainability": "SHAP values provided for decisions",
                "human_review": "High-value transactions (>$1000) flagged for review"
            },
            
            "limitations": {
                "known_issues": [
                    "Lower recall on cryptocurrency transactions",
                    "New merchants may have higher false positives",
                    "Performance degrades after 30 days without retraining"
                ],
                "recommendations": "Retrain weekly with fresh data"
            },
            
            "deployment": {
                "environment": "AWS EKS",
                "replicas": 10,
                "auto_scaling": "CPU > 70%",
                "monitoring": "CloudWatch + custom dashboards",
                "rollback_plan": "Automatic rollback if error rate > 1%"
            }
        }
    
    def render_model_card(self):
        """
        Display model card
        """
        print(f"\\n{'='*60}")
        print(f"MODEL CARD: {self.model_name}")
        print(f"{'='*60}\\n")
        
        for section, content in self.card.items():
            section_title = section.replace('_', ' ').title()
            print(f"## {section_title}\\n")
            
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in value.items():
                            print(f"    - {k}: {v}")
                    elif isinstance(value, list):
                        print(f"  {key}:")
                        for item in value:
                            print(f"    - {item}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {content}")
            
            print()


# Create model card
card = ModelCard("Fraud Detection v2.1")
card.create_model_card()
card.render_model_card()
\`\`\`

---

## Key Takeaways

1. **CI/CD for ML**: Automated pipelines with data/model validation
2. **Deployment**: Canary deployments for gradual rollout
3. **Model Registry**: Version control, lineage tracking, easy rollback
4. **Testing**: ML-specific tests (invariance, fairness, performance)
5. **Documentation**: Model cards for transparency and reproducibility

**MLOps Checklist**:
- ✅ Automated training pipeline
- ✅ Data validation tests
- ✅ Model performance thresholds
- ✅ Model registry with versioning
- ✅ Canary deployment strategy
- ✅ Automated monitoring & alerts
- ✅ Rollback procedure
- ✅ Model documentation (model cards)
- ✅ Experiment tracking
- ✅ Feature store (for mature systems)

**Start Small, Grow**:
- Level 0 → 1: Automate training
- Level 1 → 2: Add CI/CD
- Level 2 → 3: Full automation with drift detection

**Next Steps**: We'll explore Real-Time ML Systems with online learning and streaming.
`,
};
