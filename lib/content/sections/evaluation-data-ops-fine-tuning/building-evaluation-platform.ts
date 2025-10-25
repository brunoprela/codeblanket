/**
 * Building an Evaluation Platform Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const buildingEvaluationPlatform = {
  id: 'building-evaluation-platform',
  title: 'Building an Evaluation Platform',
  content: `# Building an Evaluation Platform

Master building complete production-ready evaluation platforms for AI systems.

## Overview: The Complete Evaluation Platform

A production evaluation platform needs:

1. **Dataset Management**: Store, version, and manage test sets
2. **Experiment Tracking**: Track all evaluation runs
3. **Metrics Library**: Reusable evaluation metrics
4. **CI/CD Integration**: Automatic evaluation on changes
5. **Dashboards**: Visualize results and trends
6. **Alerting**: Notify team of regressions
7. **Human Review**: Collect and manage human judgments

## Platform Architecture

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

@dataclass
class EvaluationRun:
    """Single evaluation run."""
    run_id: str
    model_name: str
    dataset_name: str
    timestamp: datetime
    metrics: Dict[str, float]
    config: Dict[str, Any]
    status: str  # "running", "completed", "failed"

class EvaluationPlatform:
    """Complete evaluation platform."""
    
    def __init__(self, storage_path: str = "./eval_platform"):
        self.storage_path = storage_path
        self.datasets: Dict[str, List[Dict]] = {}
        self.runs: List[EvaluationRun] = []
        self.metrics_library: Dict[str, callable] = {}
        
        # Initialize storage
        import os
        os.makedirs (storage_path, exist_ok=True)
        os.makedirs (f"{storage_path}/datasets", exist_ok=True)
        os.makedirs (f"{storage_path}/runs", exist_ok=True)
    
    # === Dataset Management ===
    
    def register_dataset(
        self,
        name: str,
        examples: List[Dict],
        description: str,
        version: str = "1.0"
    ):
        """Register evaluation dataset."""
        
        dataset = {
            'name': name,
            'version': version,
            'description': description,
            'examples': examples,
            'created_at': datetime.now().isoformat(),
            'num_examples': len (examples)
        }
        
        # Save to disk
        path = f"{self.storage_path}/datasets/{name}_v{version}.json"
        with open (path, 'w') as f:
            json.dump (dataset, f, indent=2)
        
        self.datasets[f"{name}_v{version}"] = examples
        
        print(f"✅ Registered dataset: {name} v{version} ({len (examples)} examples)")
    
    def get_dataset (self, name: str, version: str = "1.0") -> List[Dict]:
        """Load dataset."""
        key = f"{name}_v{version}"
        
        if key in self.datasets:
            return self.datasets[key]
        
        # Load from disk
        path = f"{self.storage_path}/datasets/{key}.json"
        with open (path, 'r') as f:
            dataset = json.load (f)
        
        self.datasets[key] = dataset['examples']
        return dataset['examples']
    
    # === Metrics Library ===
    
    def register_metric (self, name: str, metric_fn: callable):
        """Register evaluation metric."""
        self.metrics_library[name] = metric_fn
        print(f"✅ Registered metric: {name}")
    
    def get_metric (self, name: str) -> callable:
        """Get metric function."""
        if name not in self.metrics_library:
            raise ValueError (f"Metric {name} not found")
        return self.metrics_library[name]
    
    # === Experiment Tracking ===
    
    async def run_evaluation(
        self,
        model_fn: callable,
        dataset_name: str,
        metrics: List[str],
        config: Dict[str, Any] = None
    ) -> EvaluationRun:
        """Run evaluation experiment."""
        
        import uuid
        
        # Create run
        run_id = str (uuid.uuid4())[:8]
        run = EvaluationRun(
            run_id=run_id,
            model_name=config.get('model_name', 'unknown') if config else 'unknown',
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            metrics={},
            config=config or {},
            status="running"
        )
        
        print(f"\\n🚀 Starting evaluation run: {run_id}")
        print(f"   Model: {run.model_name}")
        print(f"   Dataset: {dataset_name}")
        
        try:
            # Load dataset
            dataset = self.get_dataset (dataset_name)
            
            # Run evaluation
            results = []
            for i, example in enumerate (dataset):
                if i % 10 == 0:
                    print(f"   Progress: {i}/{len (dataset)}")
                
                # Get model output
                output = await model_fn (example['input'])
                
                # Calculate metrics
                scores = {}
                for metric_name in metrics:
                    metric_fn = self.get_metric (metric_name)
                    score = metric_fn (output, example.get('expected_output'))
                    scores[metric_name] = score
                
                results.append (scores)
            
            # Aggregate metrics
            for metric_name in metrics:
                values = [r[metric_name] for r in results]
                run.metrics[metric_name] = sum (values) / len (values)
            
            run.status = "completed"
            
            print(f"\\n✅ Evaluation complete: {run_id}")
            for metric, value in run.metrics.items():
                print(f"   {metric}: {value:.2%}")
        
        except Exception as e:
            run.status = "failed"
            run.metrics['error'] = str (e)
            print(f"\\n❌ Evaluation failed: {e}")
        
        # Save run
        self.runs.append (run)
        self._save_run (run)
        
        return run
    
    def _save_run (self, run: EvaluationRun):
        """Save run to disk."""
        path = f"{self.storage_path}/runs/{run.run_id}.json"
        
        run_data = {
            'run_id': run.run_id,
            'model_name': run.model_name,
            'dataset_name': run.dataset_name,
            'timestamp': run.timestamp.isoformat(),
            'metrics': run.metrics,
            'config': run.config,
            'status': run.status
        }
        
        with open (path, 'w') as f:
            json.dump (run_data, f, indent=2)
    
    # === Comparison & Analysis ===
    
    def compare_runs(
        self,
        run_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple evaluation runs."""
        
        runs = [r for r in self.runs if r.run_id in run_ids]
        
        if not runs:
            return {'error': 'No runs found'}
        
        # Get all metrics
        all_metrics = set()
        for run in runs:
            all_metrics.update (run.metrics.keys())
        
        # Build comparison table
        comparison = {
            'runs': [
                {
                    'run_id': run.run_id,
                    'model': run.model_name,
                    'timestamp': run.timestamp.isoformat(),
                    'metrics': run.metrics
                }
                for run in runs
            ],
            'best_per_metric': {}
        }
        
        # Find best run for each metric
        for metric in all_metrics:
            values = [(run.run_id, run.metrics.get (metric, 0)) for run in runs]
            best = max (values, key=lambda x: x[1])
            comparison['best_per_metric'][metric] = {
                'run_id': best[0],
                'value': best[1]
            }
        
        return comparison
    
    def get_leaderboard(
        self,
        dataset_name: str,
        metric: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Get leaderboard for dataset and metric."""
        
        # Filter runs for this dataset
        relevant_runs = [
            r for r in self.runs
            if r.dataset_name == dataset_name and r.status == "completed"
        ]
        
        # Sort by metric
        sorted_runs = sorted(
            relevant_runs,
            key=lambda r: r.metrics.get (metric, 0),
            reverse=True
        )
        
        leaderboard = [
            {
                'rank': i + 1,
                'model': run.model_name,
                'score': run.metrics.get (metric, 0),
                'run_id': run.run_id,
                'timestamp': run.timestamp.isoformat()
            }
            for i, run in enumerate (sorted_runs[:top_k])
        ]
        
        return leaderboard
    
    # === CI/CD Integration ===
    
    def ci_evaluation_check(
        self,
        model_fn: callable,
        baseline_run_id: str,
        regression_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        CI/CD gate: Ensure no regression vs baseline.
        
        Returns pass/fail for CI pipeline.
        """
        
        # Get baseline run
        baseline = next((r for r in self.runs if r.run_id == baseline_run_id), None)
        if not baseline:
            return {'error': f'Baseline run {baseline_run_id} not found', 'pass': False}
        
        # Run evaluation with same config
        import asyncio
        new_run = asyncio.run (self.run_evaluation(
            model_fn=model_fn,
            dataset_name=baseline.dataset_name,
            metrics=list (baseline.metrics.keys()),
            config=baseline.config
        ))
        
        # Compare
        regressions = []
        for metric, baseline_value in baseline.metrics.items():
            new_value = new_run.metrics.get (metric, 0)
            
            if baseline_value > 0:
                change = (new_value - baseline_value) / baseline_value
                
                if change < -regression_threshold:
                    regressions.append({
                        'metric': metric,
                        'baseline': baseline_value,
                        'new': new_value,
                        'change_pct': change * 100
                    })
        
        passed = len (regressions) == 0
        
        return {
            'pass': passed,
            'baseline_run_id': baseline_run_id,
            'new_run_id': new_run.run_id,
            'regressions': regressions
        }
    
    # === Dashboard Data ===
    
    def get_dashboard_data (self) -> Dict[str, Any]:
        """Generate data for dashboard."""
        
        return {
            'total_runs': len (self.runs),
            'total_datasets': len (self.datasets),
            'recent_runs': [
                {
                    'run_id': r.run_id,
                    'model': r.model_name,
                    'dataset': r.dataset_name,
                    'status': r.status,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in sorted (self.runs, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            'metrics_tracked': list (self.metrics_library.keys())
        }

# === Usage Example ===

# Initialize platform
platform = EvaluationPlatform (storage_path="./my_eval_platform")

# Register datasets
platform.register_dataset(
    name="summarization_test",
    examples=test_examples,
    description="Summarization test set v1",
    version="1.0"
)

# Register metrics
def accuracy_metric (output: str, expected: str) -> float:
    return 1.0 if output.strip() == expected.strip() else 0.0

platform.register_metric("accuracy", accuracy_metric)

# Run evaluation
run = await platform.run_evaluation(
    model_fn=my_model,
    dataset_name="summarization_test",
    metrics=["accuracy"],
    config={'model_name': 'gpt-3.5-turbo', 'temperature': 0.7}
)

# Compare runs
comparison = platform.compare_runs([run1_id, run2_id, run3_id])

# Get leaderboard
leaderboard = platform.get_leaderboard(
    dataset_name="summarization_test",
    metric="accuracy",
    top_k=10
)

# CI/CD check
ci_result = platform.ci_evaluation_check(
    model_fn=new_model,
    baseline_run_id=baseline_run_id,
    regression_threshold=0.05
)

if not ci_result['pass']:
    print("❌ CI check failed - performance regression detected!")
    sys.exit(1)
\`\`\`

## Web Dashboard

\`\`\`python
# Simple Streamlit dashboard
import streamlit as st
import plotly.express as px

def render_dashboard (platform: EvaluationPlatform):
    """Render evaluation dashboard."""
    
    st.title("AI Evaluation Platform")
    
    # Overview
    data = platform.get_dashboard_data()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", data['total_runs'])
    col2.metric("Datasets", data['total_datasets'])
    col3.metric("Metrics", len (data['metrics_tracked']))
    
    # Recent runs
    st.subheader("Recent Evaluations")
    st.table (data['recent_runs'])
    
    # Leaderboard
    st.subheader("Leaderboard")
    dataset = st.selectbox("Dataset", list (platform.datasets.keys()))
    metric = st.selectbox("Metric", data['metrics_tracked'])
    
    leaderboard = platform.get_leaderboard (dataset, metric, top_k=10)
    st.table (leaderboard)
    
    # Trends over time
    st.subheader("Performance Trends")
    
    # Plot metric over time
    runs_data = [
        {
            'timestamp': r.timestamp,
            'metric_value': r.metrics.get (metric, 0),
            'model': r.model_name
        }
        for r in platform.runs
        if r.status == "completed"
    ]
    
    if runs_data:
        fig = px.line(
            runs_data,
            x='timestamp',
            y='metric_value',
            color='model',
            title=f'{metric} over time'
        )
        st.plotly_chart (fig)

# Run dashboard
# streamlit run dashboard.py
render_dashboard (platform)
\`\`\`

## Integration with CI/CD

\`\`\`yaml
# .github/workflows/evaluation.yml
name: Model Evaluation

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Evaluation
        env:
          OPENAI_API_KEY: \${{ secrets.OPENAI_API_KEY }}
        run: |
          python scripts/run_evaluation.py \\
            --baseline-run-id \${{ secrets.BASELINE_RUN_ID }} \\
            --regression-threshold 0.05
      
      - name: Comment Results
        if: always()
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse (fs.readFileSync('eval_results.json'));
            
            let comment = '## Evaluation Results\\n\\n';
            
            if (results.pass) {
              comment += '✅ No performance regression detected\\n\\n';
            } else {
              comment += '❌ Performance regression detected!\\n\\n';
              comment += '| Metric | Baseline | New | Change |\\n';
              comment += '|--------|----------|-----|--------|\\n';
              
              for (const reg of results.regressions) {
                comment += \`| \${reg.metric} | \${reg.baseline.toFixed(2)} | \${reg.new.toFixed(2)} | \${reg.change_pct.toFixed(1)}% |\\n\`;
              }
            }
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
\`\`\`

## Production Checklist

✅ **Platform Setup**
- [ ] Storage/database configured
- [ ] Dataset management system
- [ ] Metrics library created
- [ ] Experiment tracking enabled

✅ **CI/CD Integration**
- [ ] Automated evaluation on PR
- [ ] Regression detection
- [ ] Results posted to PR
- [ ] Deployment gates configured

✅ **Monitoring**
- [ ] Dashboard deployed
- [ ] Leaderboards visible
- [ ] Trend analysis available
- [ ] Alerting configured

✅ **Team Adoption**
- [ ] Documentation written
- [ ] Team trained
- [ ] Workflow established
- [ ] Regular reviews scheduled

## Congratulations!

You've completed Module 16: Evaluation, Data Operations & Fine-Tuning!

You can now:
✅ Build comprehensive evaluation frameworks
✅ Implement A/B testing for prompts and models
✅ Create and manage evaluation datasets
✅ Collect and analyze human feedback
✅ Set up data labeling pipelines
✅ Generate synthetic training data
✅ Fine-tune OpenAI and open-source models
✅ Evaluate RAG systems comprehensively
✅ Assess multi-modal outputs
✅ Monitor production systems continuously
✅ Build complete evaluation platforms
✅ Integrate evaluation into CI/CD

**Next:** Apply these skills to build production-ready AI applications with confidence!
`,
};
