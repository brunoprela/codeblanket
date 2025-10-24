/**
 * Human Evaluation & Feedback Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const humanEvaluationFeedback = {
  id: 'human-evaluation-feedback',
  title: 'Human Evaluation & Feedback',
  content: `# Human Evaluation & Feedback

Master collecting, managing, and leveraging human judgments for AI system evaluation.

## Overview: Why Human Evaluation Matters

**Humans remain the gold standard for evaluating AI quality.**

Automated metrics can't capture:
- ❌ Subtle quality nuances
- ❌ Creativity and originality
- ❌ Cultural appropriateness
- ❌ Emotional resonance
- ❌ User satisfaction
- ❌ Safety edge cases

Human evaluation provides:
- ✅ Ground truth labels
- ✅ Subjective quality assessment
- ✅ Edge case discovery
- ✅ User experience insights
- ✅ Safety validation

## When to Use Human Evaluation

\`\`\`python
def should_use_human_eval(scenario: Dict[str, Any]) -> bool:
    """Decide if human evaluation is needed."""
    
    # Always use human eval for:
    if scenario.get('safety_critical'):
        return True  # Medical, legal, safety systems
    
    if scenario.get('subjective_quality'):
        return True  # Creative writing, summarization
    
    if scenario.get('no_ground_truth'):
        return True  # Open-ended generation
    
    if scenario.get('new_domain'):
        return True  # Initial dataset creation
    
    # Sometimes use:
    if scenario.get('model_development_phase'):
        return True  # For baseline datasets
    
    # Can skip if:
    if scenario.get('objective_metrics_sufficient'):
        return False  # Code execution, math problems
    
    if scenario.get('continuous_monitoring') and scenario.get('good_proxies'):
        return False  # Use automatic metrics + occasional spot checks
    
    return False  # Default: automated metrics

# Examples
print(should_use_human_eval({
    'task': 'medical_diagnosis',
    'safety_critical': True
}))  # True

print(should_use_human_eval({
    'task': 'arithmetic',
    'objective_metrics_sufficient': True
}))  # False
\`\`\`

## Human Evaluation Workflow

### 1. Design Evaluation Protocol

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class RatingScale(Enum):
    """Common rating scales."""
    BINARY = "binary"  # Good/Bad, Pass/Fail
    LIKERT_3 = "likert_3"  # Poor, Fair, Good
    LIKERT_5 = "likert_5"  # 1-5 stars
    LIKERT_7 = "likert_7"  # 1-7 scale
    LIKERT_10 = "likert_10"  # 1-10 scale

@dataclass
class EvaluationCriterion:
    """Single evaluation criterion."""
    name: str
    description: str
    scale: RatingScale
    examples: Dict[int, str]  # Rating -> Example description
    
@dataclass
class EvaluationProtocol:
    """Complete evaluation protocol."""
    task_name: str
    instructions: str
    criteria: List[EvaluationCriterion]
    examples: List[Dict[str, Any]]  # Example evaluations
    training_examples: List[Dict[str, Any]]
    
    def to_annotation_task(self) -> Dict[str, Any]:
        """Convert to annotation platform format."""
        return {
            'task_name': self.task_name,
            'instructions': self.instructions,
            'criteria': [
                {
                    'name': c.name,
                    'description': c.description,
                    'scale': c.scale.value,
                    'examples': c.examples
                }
                for c in self.criteria
            ]
        }

# Example: Summarization evaluation protocol
summarization_protocol = EvaluationProtocol(
    task_name="summarization_quality",
    instructions="""
    Evaluate the quality of AI-generated summaries.
    Read the original article and the summary, then rate each criterion.
    """,
    criteria=[
        EvaluationCriterion(
            name="accuracy",
            description="Does the summary contain accurate information from the article?",
            scale=RatingScale.LIKERT_5,
            examples={
                1: "Major factual errors",
                3: "Mostly accurate with minor errors",
                5: "Perfectly accurate"
            }
        ),
        EvaluationCriterion(
            name="completeness",
            description="Does the summary cover all key points?",
            scale=RatingScale.LIKERT_5,
            examples={
                1: "Misses most important points",
                3: "Covers main points but misses some details",
                5: "Comprehensive coverage of all key points"
            }
        ),
        EvaluationCriterion(
            name="conciseness",
            description="Is the summary appropriately concise?",
            scale=RatingScale.LIKERT_5,
            examples={
                1: "Too verbose or too short",
                3: "Acceptable length",
                5: "Perfect length and conciseness"
            }
        ),
        EvaluationCriterion(
            name="coherence",
            description="Is the summary well-written and coherent?",
            scale=RatingScale.LIKERT_5,
            examples={
                1: "Hard to follow, poor structure",
                3: "Readable but could be clearer",
                5: "Excellent clarity and flow"
            }
        )
    ],
    examples=[],  # Add calibration examples
    training_examples=[]  # Add training examples
)
\`\`\`

### 2. Recruit and Train Evaluators

\`\`\`python
class EvaluatorManager:
    """Manage human evaluators."""
    
    def __init__(self, protocol: EvaluationProtocol):
        self.protocol = protocol
        self.evaluators: Dict[str, Dict[str, Any]] = {}
    
    def add_evaluator(
        self,
        evaluator_id: str,
        expertise_level: str,
        demographics: Dict[str, Any] = None
    ):
        """Register new evaluator."""
        self.evaluators[evaluator_id] = {
            'id': evaluator_id,
            'expertise_level': expertise_level,  # novice, intermediate, expert
            'demographics': demographics or {},
            'completed_training': False,
            'calibration_score': None,
            'evaluations_completed': 0,
            'agreement_with_gold': []
        }
    
    async def run_training(self, evaluator_id: str) -> Dict[str, Any]:
        """Run training for evaluator."""
        print(f"\\n=== Training for {evaluator_id} ===\\n")
        print(self.protocol.instructions)
        print()
        
        # Show examples
        print("Here are example evaluations:\\n")
        for i, example in enumerate(self.protocol.training_examples[:3]):
            print(f"Example {i+1}:")
            print(f"Input: {example['input'][:100]}...")
            print(f"Output: {example['output'][:100]}...")
            print("Expert ratings:")
            for criterion, rating in example['expert_ratings'].items():
                print(f"  {criterion}: {rating}/5")
            print()
        
        # Calibration test
        print("Now try rating these examples yourself:\\n")
        calibration_score = await self._calibration_test(evaluator_id)
        
        self.evaluators[evaluator_id]['completed_training'] = True
        self.evaluators[evaluator_id]['calibration_score'] = calibration_score
        
        return {
            'evaluator_id': evaluator_id,
            'passed_training': calibration_score >= 0.7,
            'calibration_score': calibration_score
        }
    
    async def _calibration_test(self, evaluator_id: str) -> float:
        """Test evaluator on gold-labeled examples."""
        # In production, present calibration examples
        # and compare evaluator ratings with gold labels
        
        # Placeholder: simulate calibration score
        import random
        return random.uniform(0.6, 0.95)
    
    def get_qualified_evaluators(
        self,
        min_calibration: float = 0.75
    ) -> List[str]:
        """Get list of qualified evaluators."""
        return [
            evaluator_id
            for evaluator_id, data in self.evaluators.items()
            if data['completed_training'] 
            and data['calibration_score'] 
            and data['calibration_score'] >= min_calibration
        ]

# Usage
manager = EvaluatorManager(summarization_protocol)

# Recruit evaluators
manager.add_evaluator("alice", "expert", {"background": "journalism"})
manager.add_evaluator("bob", "intermediate", {"background": "general"})
manager.add_evaluator("charlie", "expert", {"background": "editing"})

# Train them
for evaluator_id in ["alice", "bob", "charlie"]:
    result = await manager.run_training(evaluator_id)
    if result['passed_training']:
        print(f"✅ {evaluator_id} passed training")
    else:
        print(f"❌ {evaluator_id} needs more training")
\`\`\`

### 3. Collect Evaluations

\`\`\`python
@dataclass
class HumanEvaluation:
    """Single human evaluation."""
    evaluation_id: str
    evaluator_id: str
    example_id: str
    input: str
    output: str
    ratings: Dict[str, int]  # criterion -> rating
    confidence: Optional[int] = None  # 1-5 how confident
    comments: Optional[str] = None
    time_spent_seconds: Optional[float] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class EvaluationCollector:
    """Collect human evaluations."""
    
    def __init__(self, protocol: EvaluationProtocol):
        self.protocol = protocol
        self.evaluations: List[HumanEvaluation] = []
    
    async def collect_evaluation(
        self,
        evaluator_id: str,
        example: Dict[str, Any],
        mode: str = "terminal"  # "terminal" or "web"
    ) -> HumanEvaluation:
        """Collect single evaluation."""
        
        if mode == "terminal":
            return await self._collect_terminal(evaluator_id, example)
        else:
            # In production: web interface
            return await self._collect_web(evaluator_id, example)
    
    async def _collect_terminal(
        self,
        evaluator_id: str,
        example: Dict[str, Any]
    ) -> HumanEvaluation:
        """Collect evaluation via terminal (for demo)."""
        
        print(f"\\n{'='*60}")
        print(f"Evaluator: {evaluator_id}")
        print(f"{'='*60}\\n")
        
        print("INPUT:")
        print(example['input'])
        print("\\nOUTPUT:")
        print(example['output'])
        print()
        
        ratings = {}
        start_time = time.time()
        
        for criterion in self.protocol.criteria:
            print(f"\\n{criterion.name.upper()}: {criterion.description}")
            print(f"Scale: {criterion.scale.value}")
            
            # Show examples
            if criterion.examples:
                print("Examples:")
                for rating, description in sorted(criterion.examples.items()):
                    print(f"  {rating}: {description}")
            
            # Get rating
            while True:
                try:
                    rating = int(input(f"\\nRating (1-5): "))
                    if 1 <= rating <= 5:
                        ratings[criterion.name] = rating
                        break
                    else:
                        print("Please enter a number between 1 and 5")
                except ValueError:
                    print("Please enter a valid number")
        
        confidence = int(input("\\nOverall confidence in your ratings (1-5): "))
        comments = input("Comments (optional): ")
        
        time_spent = time.time() - start_time
        
        evaluation = HumanEvaluation(
            evaluation_id=f"eval_{int(time.time())}",
            evaluator_id=evaluator_id,
            example_id=example.get('id', 'unknown'),
            input=example['input'],
            output=example['output'],
            ratings=ratings,
            confidence=confidence,
            comments=comments or None,
            time_spent_seconds=time_spent
        )
        
        self.evaluations.append(evaluation)
        return evaluation
    
    async def _collect_web(
        self,
        evaluator_id: str,
        example: Dict[str, Any]
    ) -> HumanEvaluation:
        """Collect via web interface."""
        # In production: integrate with Label Studio, Prodigy, Scale AI, etc.
        pass
    
    def export_evaluations(self, output_path: str):
        """Export collected evaluations."""
        import json
        
        data = [
            {
                'evaluation_id': e.evaluation_id,
                'evaluator_id': e.evaluator_id,
                'example_id': e.example_id,
                'ratings': e.ratings,
                'confidence': e.confidence,
                'comments': e.comments,
                'time_spent_seconds': e.time_spent_seconds,
                'timestamp': e.timestamp
            }
            for e in self.evaluations
        ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {len(data)} evaluations to {output_path}")

# Usage
collector = EvaluationCollector(summarization_protocol)

for example in examples_to_evaluate:
    evaluation = await collector.collect_evaluation(
        evaluator_id="alice",
        example=example,
        mode="terminal"
    )

collector.export_evaluations("evaluations/human_eval_round1.json")
\`\`\`

## Inter-Annotator Agreement

### Measuring Agreement

\`\`\`python
from scipy.stats import cohen_kappa_score
import numpy as np

class AgreementAnalyzer:
    """Analyze inter-annotator agreement."""
    
    def calculate_cohens_kappa(
        self,
        evaluator1_ratings: List[int],
        evaluator2_ratings: List[int]
    ) -> float:
        """
        Cohen's Kappa: Agreement between two evaluators.
        
        Interpretation:
        < 0: No agreement
        0-0.20: Slight
        0.21-0.40: Fair
        0.41-0.60: Moderate
        0.61-0.80: Substantial
        0.81-1.00: Almost perfect
        """
        return cohen_kappa_score(evaluator1_ratings, evaluator2_ratings)
    
    def calculate_fleiss_kappa(
        self,
        ratings_matrix: np.ndarray
    ) -> float:
        """
        Fleiss' Kappa: Agreement among multiple evaluators.
        
        ratings_matrix shape: (n_examples, n_evaluators)
        """
        n_examples, n_evaluators = ratings_matrix.shape
        n_categories = int(ratings_matrix.max()) - int(ratings_matrix.min()) + 1
        
        # Calculate agreement
        # (Implementation of Fleiss' Kappa formula)
        # Placeholder for brevity
        return 0.75  # Example score
    
    def analyze_agreement(
        self,
        evaluations: List[HumanEvaluation],
        criterion: str
    ) -> Dict[str, Any]:
        """Analyze agreement for specific criterion."""
        
        # Group by example
        by_example = {}
        for eval in evaluations:
            if eval.example_id not in by_example:
                by_example[eval.example_id] = []
            by_example[eval.example_id].append({
                'evaluator_id': eval.evaluator_id,
                'rating': eval.ratings.get(criterion)
            })
        
        # Filter examples with multiple evaluators
        multi_evaluated = {
            example_id: ratings
            for example_id, ratings in by_example.items()
            if len(ratings) >= 2
        }
        
        # Calculate pairwise agreement
        agreements = []
        for example_id, ratings in multi_evaluated.items():
            rating_values = [r['rating'] for r in ratings]
            
            # Variance (lower = more agreement)
            if len(rating_values) > 1:
                variance = np.var(rating_values)
                agreements.append(variance)
        
        avg_variance = np.mean(agreements) if agreements else 0
        
        # Calculate kappa for pairs
        if len(multi_evaluated) >= 10:
            # Get two most active evaluators
            evaluator_counts = {}
            for eval in evaluations:
                evaluator_counts[eval.evaluator_id] = evaluator_counts.get(eval.evaluator_id, 0) + 1
            
            top_evaluators = sorted(evaluator_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            
            if len(top_evaluators) == 2:
                eval1_id, eval2_id = top_evaluators[0][0], top_evaluators[1][0]
                
                # Get their ratings on same examples
                eval1_ratings = []
                eval2_ratings = []
                
                for eval in evaluations:
                    if eval.evaluator_id == eval1_id and criterion in eval.ratings:
                        # Find matching evaluation from eval2
                        matching = [
                            e for e in evaluations
                            if e.evaluator_id == eval2_id and e.example_id == eval.example_id
                        ]
                        if matching:
                            eval1_ratings.append(eval.ratings[criterion])
                            eval2_ratings.append(matching[0].ratings[criterion])
                
                if len(eval1_ratings) >= 10:
                    kappa = self.calculate_cohens_kappa(eval1_ratings, eval2_ratings)
                else:
                    kappa = None
            else:
                kappa = None
        else:
            kappa = None
        
        return {
            'criterion': criterion,
            'examples_with_multiple_evals': len(multi_evaluated),
            'average_variance': avg_variance,
            'cohens_kappa': kappa,
            'interpretation': self._interpret_kappa(kappa) if kappa else "insufficient_data"
        }
    
    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret kappa score."""
        if kappa < 0:
            return "no_agreement"
        elif kappa < 0.20:
            return "slight"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "almost_perfect"

# Usage
analyzer = AgreementAnalyzer()

agreement = analyzer.analyze_agreement(evaluations, criterion="accuracy")
print(f"Agreement for accuracy:")
print(f"  Examples evaluated by multiple people: {agreement['examples_with_multiple_evals']}")
print(f"  Average variance: {agreement['average_variance']:.2f}")
if agreement['cohens_kappa']:
    print(f"  Cohen's Kappa: {agreement['cohens_kappa']:.2f} ({agreement['interpretation']})")
\`\`\`

## Integrating with Annotation Platforms

### Label Studio Integration

\`\`\`python
import requests

class LabelStudioIntegration:
    """Integrate with Label Studio annotation platform."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {'Authorization': f'Token {api_key}'}
    
    def create_project(
        self,
        protocol: EvaluationProtocol
    ) -> int:
        """Create Label Studio project."""
        
        # Generate Label Studio config
        config = self._generate_labeling_config(protocol)
        
        project_data = {
            'title': protocol.task_name,
            'description': protocol.instructions,
            'label_config': config
        }
        
        response = requests.post(
            f"{self.api_url}/api/projects",
            headers=self.headers,
            json=project_data
        )
        
        project_id = response.json()['id']
        return project_id
    
    def upload_tasks(
        self,
        project_id: int,
        examples: List[Dict[str, Any]]
    ):
        """Upload examples to Label Studio."""
        
        tasks = [
            {
                'data': {
                    'input': ex['input'],
                    'output': ex['output'],
                    'example_id': ex.get('id', str(i))
                }
            }
            for i, ex in enumerate(examples)
        ]
        
        response = requests.post(
            f"{self.api_url}/api/projects/{project_id}/import",
            headers=self.headers,
            json=tasks
        )
        
        return response.json()
    
    def export_annotations(self, project_id: int) -> List[Dict]:
        """Export completed annotations."""
        
        response = requests.get(
            f"{self.api_url}/api/projects/{project_id}/export",
            headers=self.headers,
            params={'exportType': 'JSON'}
        )
        
        return response.json()
    
    def _generate_labeling_config(
        self,
        protocol: EvaluationProtocol
    ) -> str:
        """Generate Label Studio XML config."""
        
        config = '<View>\\n'
        
        # Display input and output
        config += '  <Text name="input" value="$input"/>\\n'
        config += '  <Text name="output" value="$output"/>\\n'
        
        # Add rating controls for each criterion
        for criterion in protocol.criteria:
            if criterion.scale == RatingScale.LIKERT_5:
                config += f'  <Rating name="{criterion.name}" toName="output" maxRating="5" icon="star"/>\\n'
            else:
                config += f'  <Choices name="{criterion.name}" toName="output" choice="single">\\n'
                for rating in range(1, 6):
                    config += f'    <Choice value="{rating}"/>\\n'
                config += '  </Choices>\\n'
        
        # Comments
        config += '  <TextArea name="comments" toName="output" placeholder="Comments (optional)"/>\\n'
        
        config += '</View>'
        
        return config

# Usage
label_studio = LabelStudioIntegration(
    api_url="http://localhost:8080",
    api_key="your_api_key"
)

project_id = label_studio.create_project(summarization_protocol)
label_studio.upload_tasks(project_id, examples_to_evaluate)

# Later, export annotations
annotations = label_studio.export_annotations(project_id)
\`\`\`

## Production Checklist

✅ **Protocol Design**
- [ ] Clear evaluation criteria defined
- [ ] Appropriate rating scales chosen
- [ ] Comprehensive instructions written
- [ ] Calibration examples prepared
- [ ] Edge cases included

✅ **Evaluator Management**
- [ ] Training materials created
- [ ] Qualification tests designed
- [ ] Inter-annotator agreement monitored
- [ ] Feedback mechanism for evaluators

✅ **Quality Control**
- [ ] Gold standard examples for validation
- [ ] Regular agreement checks
- [ ] Outlier detection
- [ ] Feedback loops to improve protocol

✅ **Integration**
- [ ] Annotation platform selected and configured
- [ ] Automated export/import workflows
- [ ] Version control for evaluation data
- [ ] Analytics dashboard

## Next Steps

You now understand human evaluation. Next, learn:
- Data labeling at scale
- Synthetic data generation
- Fine-tuning fundamentals
- Continuous evaluation workflows
`,
};
