/**
 * Data Labeling & Annotation Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const dataLabelingAnnotation = {
  id: 'data-labeling-annotation',
  title: 'Data Labeling & Annotation',
  content: `# Data Labeling & Annotation

Master building production data labeling pipelines for training and evaluation datasets.

## Overview: The Data Labeling Challenge

**High-quality AI requires high-quality labeled data.**

The reality:
- 🔄 Labeling is tedious and expensive
- 👥 Requires domain expertise
- ⚖️ Consistency is hard across annotators
- 📈 Need thousands to millions of examples
- 💰 Cost can exceed $100K for large datasets

The goal:
- ✅ Scale labeling efficiently
- ✅ Maintain quality and consistency
- ✅ Minimize cost
- ✅ Accelerate iteration cycles

## Data Labeling Strategies

### 1. In-House vs Outsourced

\`\`\`python
class LabelingStrategy:
    """Choose labeling approach based on requirements."""
    
    @staticmethod
    def recommend_strategy(
        domain_complexity: str,  # "simple", "moderate", "expert"
        dataset_size: int,
        budget_per_label: float,
        timeline_weeks: int
    ) -> Dict[str, Any]:
        """Recommend labeling strategy."""
        
        if domain_complexity == "expert":
            # Need domain experts
            strategy = "in_house_experts"
            estimated_cost = dataset_size * (budget_per_label * 3)  # Experts cost more
            throughput = 50  # Labels per expert per day
            
        elif dataset_size < 1000 and timeline_weeks <= 2:
            # Small, fast
            strategy = "in_house_team"
            estimated_cost = dataset_size * budget_per_label
            throughput = 200  # Labels per person per day
            
        elif dataset_size > 10000:
            # Large scale
            strategy = "crowdsourcing_platform"
            estimated_cost = dataset_size * (budget_per_label * 0.5)  # Cheaper
            throughput = 5000  # High throughput with multiple workers
            
        else:
            # Moderate size, quality important
            strategy = "managed_service"  # Scale AI, Labelbox, etc.
            estimated_cost = dataset_size * budget_per_label * 1.5
            throughput = 1000
        
        estimated_days = dataset_size / throughput
        
        return {
            'recommended_strategy': strategy,
            'estimated_cost': estimated_cost,
            'estimated_days': estimated_days,
            'reasoning': f"Based on {domain_complexity} complexity and {dataset_size} examples"
        }

# Usage
recommendation = LabelingStrategy.recommend_strategy(
    domain_complexity="moderate",
    dataset_size=5000,
    budget_per_label=1.0,
    timeline_weeks=4
)

print(f"Recommended: {recommendation['recommended_strategy']}")
print(f"Est. cost: \${recommendation['estimated_cost']:, .2f
}")
print(f"Est. time: {recommendation['estimated_days']:.0f} days")
\`\`\`

### 2. Quality Control Mechanisms

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Any
import random

@dataclass
class LabelingTask:
    """Single labeling task."""
    task_id: str
    data: Dict[str, Any]
    assigned_to: List[str]  # Multiple annotators for agreement
    labels: Dict[str, Any]  # annotator_id -> label
    gold_standard: Any = None  # If known
    is_test_question: bool = False
    
class QualityController:
    """Implement quality control for labeling."""
    
    def __init__(
        self,
        redundancy: int = 3,  # How many annotators per task
        test_question_ratio: float = 0.1,  # % of gold standard questions
        min_accuracy_threshold: float = 0.85
    ):
        self.redundancy = redundancy
        self.test_question_ratio = test_question_ratio
        self.min_accuracy_threshold = min_accuracy_threshold
        
        self.annotator_stats: Dict[str, Dict] = {}
        self.gold_questions: List[LabelingTask] = []
    
    def create_labeling_batch(
        self,
        tasks: List[Dict[str, Any]],
        annotators: List[str]
    ) -> List[LabelingTask]:
        """Create batch with quality controls."""
        
        labeling_tasks = []
        
        # Insert test questions
        n_tests = int(len(tasks) * self.test_question_ratio)
        test_indices = random.sample(range(len(tasks)), n_tests)
        
        for i, task_data in enumerate(tasks):
            # Assign to multiple annotators
            assigned = random.sample(annotators, min(self.redundancy, len(annotators)))
            
            is_test = i in test_indices
            gold = task_data.get('gold_label') if is_test else None
            
            labeling_task = LabelingTask(
                task_id=f"task_{i}",
                data=task_data,
                assigned_to=assigned,
                labels={},
                gold_standard=gold,
                is_test_question=is_test
            )
            
            labeling_tasks.append(labeling_task)
            
            if is_test:
                self.gold_questions.append(labeling_task)
        
        return labeling_tasks
    
    def record_label(
        self,
        task: LabelingTask,
        annotator_id: str,
        label: Any
    ):
        """Record label from annotator."""
        task.labels[annotator_id] = label
        
        # Update annotator stats
        if annotator_id not in self.annotator_stats:
            self.annotator_stats[annotator_id] = {
                'total_labeled': 0,
                'test_questions_correct': 0,
                'test_questions_total': 0,
                'accuracy': None
            }
        
        stats = self.annotator_stats[annotator_id]
        stats['total_labeled'] += 1
        
        # Check against gold standard if test question
        if task.is_test_question and task.gold_standard is not None:
            stats['test_questions_total'] += 1
            if label == task.gold_standard:
                stats['test_questions_correct'] += 1
            
            # Update accuracy
            stats['accuracy'] = (
                stats['test_questions_correct'] / stats['test_questions_total']
            )
    
    def get_consensus_label(self, task: LabelingTask) -> Dict[str, Any]:
        """Get consensus from multiple annotations."""
        if len(task.labels) < 2:
            # Not enough labels yet
            return {'status': 'pending', 'label': None, 'confidence': 0}
        
        # Majority vote
        from collections import Counter
        label_counts = Counter(task.labels.values())
        most_common = label_counts.most_common(1)[0]
        majority_label = most_common[0]
        majority_count = most_common[1]
        
        # Confidence = % agreement
        confidence = majority_count / len(task.labels)
        
        # Require minimum agreement
        if confidence >= 0.67:  # 2/3 agreement
            status = 'consensus'
        else:
            status = 'dispute'  # Need more annotators or expert review
        
        return {
            'status': status,
            'label': majority_label,
            'confidence': confidence,
            'vote_distribution': dict(label_counts)
        }
    
    def flag_poor_annotators(self) -> List[str]:
        """Identify annotators below quality threshold."""
        poor_annotators = []
        
        for annotator_id, stats in self.annotator_stats.items():
            if stats['accuracy'] is not None:
                if stats['accuracy'] < self.min_accuracy_threshold:
                    poor_annotators.append(annotator_id)
        
        return poor_annotators
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate quality control report."""
        report = {
            'total_annotators': len(self.annotator_stats),
            'annotators': []
        }
        
        for annotator_id, stats in self.annotator_stats.items():
            report['annotators'].append({
                'id': annotator_id,
                'labels_completed': stats['total_labeled'],
                'accuracy': stats['accuracy'],
                'status': 'good' if not stats['accuracy'] or stats['accuracy'] >= self.min_accuracy_threshold else 'poor'
            })
        
        # Overall statistics
        if report['annotators']:
            accuracies = [a['accuracy'] for a in report['annotators'] if a['accuracy'] is not None]
            report['average_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else None
        
        return report

# Usage
qc = QualityController(redundancy=3, test_question_ratio=0.1)

# Create batch
tasks = qc.create_labeling_batch(
    tasks=unlabeled_data,
    annotators=["alice", "bob", "charlie", "david"]
)

# Simulate labeling
for task in tasks:
    for annotator in task.assigned_to:
        # In production: annotator provides label via UI
        label = get_label_from_annotator(annotator, task.data)
        qc.record_label(task, annotator, label)

# Get consensus
for task in tasks:
    consensus = qc.get_consensus_label(task)
    if consensus['status'] == 'consensus':
        final_label = consensus['label']
    else:
        # Send for expert review
        pass

# Quality report
report = qc.generate_quality_report()
print(f"Average annotator accuracy: {report['average_accuracy']:.2%}")

poor = qc.flag_poor_annotators()
if poor:
    print(f"⚠️  Poor performing annotators: {poor}")
\`\`\`

## Active Learning

Label most informative examples first:

\`\`\`python
from sentence_transformers import SentenceTransformer
import numpy as np

class ActiveLearningSelector:
    """Select most informative examples to label."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def select_uncertain_samples(
        self,
        unlabeled_data: List[str],
        model_predictions: List[Dict[str, float]],  # class -> probability
        n_samples: int = 100
    ) -> List[int]:
        """
        Select samples where model is most uncertain.
        High uncertainty = most informative to label.
        """
        uncertainties = []
        
        for i, prediction in enumerate(model_predictions):
            # Entropy as uncertainty measure
            probs = list(prediction.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            uncertainties.append((i, entropy))
        
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Return indices of top n
        return [idx for idx, _ in uncertainties[:n_samples]]
    
    def select_diverse_samples(
        self,
        unlabeled_data: List[str],
        n_samples: int = 100
    ) -> List[int]:
        """
        Select diverse samples (maximize coverage).
        Ensures labels cover different types of inputs.
        """
        # Embed all data
        embeddings = self.model.encode(unlabeled_data)
        
        # Use k-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(embeddings)
        
        # Select one example closest to each cluster center
        selected = []
        for center in kmeans.cluster_centers_:
            # Find closest point to this center
            distances = np.linalg.norm(embeddings - center, axis=1)
            closest_idx = np.argmin(distances)
            selected.append(closest_idx)
        
        return selected
    
    def select_hybrid(
        self,
        unlabeled_data: List[str],
        model_predictions: List[Dict[str, float]],
        n_samples: int = 100,
        uncertainty_weight: float = 0.5
    ) -> List[int]:
        """
        Hybrid: combine uncertainty and diversity.
        Best of both worlds.
        """
        n_uncertain = int(n_samples * uncertainty_weight)
        n_diverse = n_samples - n_uncertain
        
        # Select uncertain samples
        uncertain_indices = self.select_uncertain_samples(
            unlabeled_data,
            model_predictions,
            n_uncertain
        )
        
        # Remove uncertain samples from pool
        remaining_data = [
            text for i, text in enumerate(unlabeled_data)
            if i not in uncertain_indices
        ]
        
        # Select diverse from remaining
        diverse_indices = self.select_diverse_samples(remaining_data, n_diverse)
        
        # Combine
        selected = uncertain_indices + diverse_indices
        
        return selected

# Usage
active_learner = ActiveLearningSelector()

# You have 100K unlabeled examples
unlabeled_data = load_unlabeled_data()
model_predictions = get_model_predictions(unlabeled_data)

# Select 1000 most informative to label
selected_indices = active_learner.select_hybrid(
    unlabeled_data,
    model_predictions,
    n_samples=1000,
    uncertainty_weight=0.6  # 60% uncertain, 40% diverse
)

# Label only these selected examples
to_label = [unlabeled_data[i] for i in selected_indices]
print(f"Selected {len(to_label)} examples for labeling")
print("This should give 80% of the benefit at 1% of the cost!")
\`\`\`

## Weak Supervision

Use heuristics and rules to generate noisy labels:

\`\`\`python
from typing import Callable, Optional

class LabelingFunction:
    """Single labeling function (heuristic)."""
    
    def __init__(
        self,
        name: str,
        function: Callable,
        accuracy: float = 0.7  # Estimated accuracy
    ):
        self.name = name
        self.function = function
        self.accuracy = accuracy
    
    def apply(self, data: Any) -> Optional[int]:
        """Apply function, return label or None if abstains."""
        try:
            return self.function(data)
        except:
            return None  # Abstain

class WeakSupervisionPipeline:
    """Combine multiple weak labeling functions."""
    
    def __init__(self, labeling_functions: List[LabelingFunction]):
        self.labeling_functions = labeling_functions
    
    def apply_all(self, data: Any) -> List[Optional[int]]:
        """Apply all labeling functions."""
        return [lf.apply(data) for lf in self.labeling_functions]
    
    def aggregate_labels(
        self,
        votes: List[Optional[int]]
    ) -> Optional[int]:
        """
        Aggregate votes from labeling functions.
        Use majority vote weighted by accuracy.
        """
        from collections import Counter
        
        # Filter out abstentions (None)
        valid_votes = [(vote, self.labeling_functions[i].accuracy)
                       for i, vote in enumerate(votes) if vote is not None]
        
        if not valid_votes:
            return None  # All abstained
        
        # Weighted voting
        weighted_votes = Counter()
        for vote, weight in valid_votes:
            weighted_votes[vote] += weight
        
        # Return majority
        return weighted_votes.most_common(1)[0][0]
    
    def label_dataset(
        self,
        dataset: List[Any]
    ) -> List[Dict[str, Any]]:
        """Label entire dataset using weak supervision."""
        
        labeled = []
        
        for i, data in enumerate(dataset):
            votes = self.apply_all(data)
            label = self.aggregate_labels(votes)
            
            if label is not None:
                labeled.append({
                    'data': data,
                    'label': label,
                    'confidence': self._estimate_confidence(votes),
                    'source': 'weak_supervision'
                })
        
        coverage = len(labeled) / len(dataset)
        print(f"Weak supervision coverage: {coverage:.1%}")
        
        return labeled
    
    def _estimate_confidence(self, votes: List[Optional[int]]) -> float:
        """Estimate confidence in aggregated label."""
        valid_votes = [v for v in votes if v is not None]
        if not valid_votes:
            return 0.0
        
        from collections import Counter
        vote_counts = Counter(valid_votes)
        most_common_count = vote_counts.most_common(1)[0][1]
        
        # Confidence = % agreement among non-abstaining functions
        return most_common_count / len(valid_votes)

# Example: Spam classification with weak supervision
def lf_contains_spam_words(email: str) -> Optional[int]:
    """Check for spam keywords."""
    spam_words = ['free', 'win', 'prize', 'click here', 'urgent']
    if any(word in email.lower() for word in spam_words):
        return 1  # Spam
    return None  # Abstain

def lf_has_suspicious_links(email: str) -> Optional[int]:
    """Check for suspicious patterns."""
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email)
    if len(urls) > 5:
        return 1  # Spam
    return None

def lf_personal_greeting(email: str) -> Optional[int]:
    """Personal emails usually not spam."""
    personal = ['hi [name]', 'hey', 'hello [name]']
    if any(greeting in email.lower() for greeting in personal):
        return 0  # Not spam
    return None

# Create pipeline
lfs = [
    LabelingFunction("spam_words", lf_contains_spam_words, accuracy=0.75),
    LabelingFunction("suspicious_links", lf_has_suspicious_links, accuracy=0.80),
    LabelingFunction("personal_greeting", lf_personal_greeting, accuracy=0.70)
]

weak_sup = WeakSupervisionPipeline(lfs)

# Label dataset
labeled_data = weak_sup.label_dataset(unlabeled_emails)

# Now train model on these noisy labels
# (Snorkel library provides advanced probabilistic aggregation)
\`\`\`

## Production Data Labeling Platform

\`\`\`python
class ProductionLabelingPlatform:
    """Complete labeling platform."""
    
    def __init__(self):
        self.quality_controller = QualityController()
        self.active_learner = ActiveLearningSelector()
        self.tasks: Dict[str, LabelingTask] = {}
    
    def create_labeling_campaign(
        self,
        data: List[Dict],
        annotators: List[str],
        use_active_learning: bool = True
    ) -> str:
        """Create new labeling campaign."""
        
        campaign_id = f"campaign_{int(time.time())}"
        
        # Select subset with active learning
        if use_active_learning and len(data) > 1000:
            # Get model predictions for uncertainty
            texts = [d['text'] for d in data]
            predictions = self._get_model_predictions(texts)
            
            selected_indices = self.active_learner.select_hybrid(
                texts, predictions, n_samples=1000
            )
            
            data = [data[i] for i in selected_indices]
        
        # Create tasks with quality controls
        tasks = self.quality_controller.create_labeling_batch(data, annotators)
        
        for task in tasks:
            self.tasks[task.task_id] = task
        
        return campaign_id
    
    async def assign_next_task(self, annotator_id: str) -> Optional[LabelingTask]:
        """Get next task for annotator."""
        for task in self.tasks.values():
            if annotator_id in task.assigned_to and annotator_id not in task.labels:
                return task
        return None
    
    async def submit_label(
        self,
        task_id: str,
        annotator_id: str,
        label: Any
    ) -> Dict[str, Any]:
        """Submit label and check quality."""
        
        task = self.tasks[task_id]
        self.quality_controller.record_label(task, annotator_id, label)
        
        # Check if task is complete
        consensus = self.quality_controller.get_consensus_label(task)
        
        response = {
            'task_id': task_id,
            'status': consensus['status'],
            'annotator_stats': self.quality_controller.annotator_stats.get(annotator_id)
        }
        
        return response
    
    def get_completed_labels(self) -> List[Dict]:
        """Get all completed, high-quality labels."""
        completed = []
        
        for task in self.tasks.values():
            consensus = self.quality_controller.get_consensus_label(task)
            
            if consensus['status'] == 'consensus':
                completed.append({
                    'data': task.data,
                    'label': consensus['label'],
                    'confidence': consensus['confidence']
                })
        
        return completed
    
    def _get_model_predictions(self, texts: List[str]) -> List[Dict]:
        """Get model predictions for active learning."""
        # Placeholder
        return [{'spam': 0.5, 'not_spam': 0.5} for _ in texts]

# Usage
platform = ProductionLabelingPlatform()

# Create campaign
campaign_id = platform.create_labeling_campaign(
    data=unlabeled_data,
    annotators=["alice", "bob", "charlie"],
    use_active_learning=True
)

# Annotators label
for annotator in ["alice", "bob", "charlie"]:
    while True:
        task = await platform.assign_next_task(annotator)
        if not task:
            break
        
        label = get_label_from_ui(annotator, task)
        result = await platform.submit_label(task.task_id, annotator, label)
        
        if result['annotator_stats']['accuracy'] and result['annotator_stats']['accuracy'] < 0.85:
            print(f"⚠️  {annotator} accuracy dropping: {result['annotator_stats']['accuracy']:.2%}")

# Export
labeled_dataset = platform.get_completed_labels()
print(f"✅ Labeled {len(labeled_dataset)} examples")
\`\`\`

## Production Checklist

✅ **Strategy**
- [ ] Labeling approach chosen based on scale and complexity
- [ ] Budget and timeline estimated
- [ ] Active learning to minimize labels needed

✅ **Quality**
- [ ] Multiple annotators per task
- [ ] Gold standard test questions
- [ ] Inter-annotator agreement monitoring
- [ ] Poor annotator detection and retraining

✅ **Efficiency**
- [ ] Annotation platform integrated
- [ ] Clear guidelines and examples
- [ ] Automated quality checks
- [ ] Feedback loops for continuous improvement

✅ **Scale**
- [ ] Weak supervision for initial labels
- [ ] Progressive refinement strategy
- [ ] Cost tracking and optimization

## Next Steps

You now understand data labeling. Next, learn:
- Synthetic data generation
- Fine-tuning fundamentals
- Continuous evaluation
- Production workflows
`,
};
