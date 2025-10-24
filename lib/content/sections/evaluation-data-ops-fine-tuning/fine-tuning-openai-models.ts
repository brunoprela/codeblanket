/**
 * Fine-Tuning OpenAI Models Section
 * Module 16: Evaluation, Data Operations & Fine-Tuning
 */

export const fineTuningOpenAIModels = {
  id: 'fine-tuning-openai-models',
  title: 'Fine-Tuning OpenAI Models',
  content: `# Fine-Tuning OpenAI Models

Master the complete workflow for fine-tuning GPT-3.5 and GPT-4 via OpenAI's API.

## Overview: OpenAI Fine-Tuning

OpenAI allows fine-tuning of:
- ✅ **gpt-3.5-turbo**: Fast, cost-effective ($0.008/1K tokens)
- ✅ **gpt-4**: Higher quality (coming soon officially)
- ✅ **davinci-002**: Legacy model

**Key Benefits:**
- No infrastructure management
- Automatic training and deployment
- Pay per use (no upfront GPU costs)
- Built-in versioning and rollback

## Complete Fine-Tuning Workflow

\`\`\`python
import openai
import json
from typing import List, Dict, Any

class OpenAIFineTuner:
    """Complete fine-tuning workflow for OpenAI models."""
    
    def __init__(self, api_key: str):
        openai.api_key = api_key
    
    def prepare_training_file(
        self,
        examples: List[Dict[str, str]],
        output_path: str = "training_data.jsonl"
    ):
        """
        Format data for OpenAI fine-tuning.
        
        Format: JSONL with messages array
        """
        with open(output_path, 'w') as f:
            for ex in examples:
                # OpenAI format
                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": ex.get('system', 'You are a helpful assistant.')
                        },
                        {
                            "role": "user",
                            "content": ex['input']
                        },
                        {
                            "role": "assistant",
                            "content": ex['output']
                        }
                    ]
                }
                f.write(json.dumps(entry) + '\\n')
        
        print(f"✅ Prepared {len(examples)} examples in {output_path}")
        return output_path
    
    async def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI."""
        
        with open(file_path, 'rb') as f:
            response = await openai.File.acreate(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response['id']
        print(f"✅ Uploaded file: {file_id}")
        return file_id
    
    async def create_fine_tune_job(
        self,
        training_file_id: str,
        validation_file_id: str = None,
        model: str = "gpt-3.5-turbo",
        n_epochs: int = 3,
        suffix: str = None
    ) -> str:
        """Create fine-tuning job."""
        
        hyperparameters = {
            "n_epochs": n_epochs
        }
        
        response = await openai.FineTuningJob.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix  # Custom model name suffix
        )
        
        job_id = response['id']
        print(f"✅ Created fine-tuning job: {job_id}")
        return job_id
    
    async def monitor_job(self, job_id: str):
        """Monitor fine-tuning progress."""
        import time
        
        print(f"\\nMonitoring job {job_id}...")
        
        while True:
            job = await openai.FineTuningJob.retrieve(job_id)
            status = job['status']
            
            print(f"Status: {status}")
            
            if status == 'succeeded':
                model_name = job['fine_tuned_model']
                print(f"\\n✅ Fine-tuning complete!")
                print(f"Model: {model_name}")
                return model_name
            
            elif status == 'failed':
                print(f"\\n❌ Fine-tuning failed!")
                print(f"Error: {job.get('error', 'Unknown error')}")
                return None
            
            elif status in ['validating_files', 'queued', 'running']:
                # Show progress
                if 'trained_tokens' in job:
                    print(f"  Trained tokens: {job['trained_tokens']}")
                
                # Wait before checking again
                await asyncio.sleep(60)  # Check every minute
            
            else:
                print(f"Unknown status: {status}")
                await asyncio.sleep(60)
    
    async def test_fine_tuned_model(
        self,
        model_name: str,
        test_examples: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Test fine-tuned model on examples."""
        
        results = []
        
        for ex in test_examples:
            response = await openai.ChatCompletion.acreate(
                model=model_name,
                messages=[
                    {"role": "system", "content": ex.get('system', 'You are a helpful assistant.')},
                    {"role": "user", "content": ex['input']}
                ],
                temperature=0.7
            )
            
            output = response.choices[0].message.content
            
            results.append({
                'input': ex['input'],
                'expected': ex['output'],
                'actual': output,
                'match': output.strip() == ex['output'].strip()
            })
        
        accuracy = sum(1 for r in results if r['match']) / len(results)
        print(f"\\nTest Accuracy: {accuracy:.2%}")
        
        return results
    
    async def complete_workflow(
        self,
        training_examples: List[Dict],
        validation_examples: List[Dict],
        test_examples: List[Dict],
        model: str = "gpt-3.5-turbo",
        epochs: int = 3
    ) -> str:
        """
        Complete end-to-end fine-tuning workflow.
        
        Returns: fine-tuned model name
        """
        
        print("=== OpenAI Fine-Tuning Workflow ===\\n")
        
        # Step 1: Prepare data
        print("Step 1: Preparing training data...")
        train_file = self.prepare_training_file(training_examples, "train.jsonl")
        val_file = self.prepare_training_file(validation_examples, "val.jsonl")
        
        # Step 2: Upload files
        print("\\nStep 2: Uploading files...")
        train_file_id = await self.upload_training_file(train_file)
        val_file_id = await self.upload_training_file(val_file)
        
        # Step 3: Create job
        print("\\nStep 3: Creating fine-tuning job...")
        job_id = await self.create_fine_tune_job(
            training_file_id=train_file_id,
            validation_file_id=val_file_id,
            model=model,
            n_epochs=epochs,
            suffix="custom-v1"
        )
        
        # Step 4: Monitor
        print("\\nStep 4: Monitoring training...")
        model_name = await self.monitor_job(job_id)
        
        if not model_name:
            raise Exception("Fine-tuning failed")
        
        # Step 5: Test
        print("\\nStep 5: Testing fine-tuned model...")
        results = await self.test_fine_tuned_model(model_name, test_examples)
        
        print(f"\\n✅ Workflow complete!")
        print(f"Model name: {model_name}")
        
        return model_name

# Usage
finetuner = OpenAIFineTuner(api_key="your-api-key")

model_name = await finetuner.complete_workflow(
    training_examples=train_data,      # 500+ examples
    validation_examples=val_data,      # 100+ examples
    test_examples=test_data,           # 50+ examples
    model="gpt-3.5-turbo",
    epochs=3
)

# Now use your fine-tuned model!
response = await openai.ChatCompletion.acreate(
    model=model_name,
    messages=[{"role": "user", "content": "Test prompt"}]
)
\`\`\`

## Cost Estimation

\`\`\`python
class FineTuningCostCalculator:
    """Calculate fine-tuning costs."""
    
    COSTS = {
        'gpt-3.5-turbo': {
            'training': 0.0080,  # per 1K tokens
            'inference': 0.0120  # per 1K tokens (input + output)
        },
        'davinci-002': {
            'training': 0.0060,
            'inference': 0.0120
        }
    }
    
    def estimate_training_cost(
        self,
        num_examples: int,
        avg_tokens_per_example: int,
        epochs: int,
        model: str = 'gpt-3.5-turbo'
    ) -> float:
        """Estimate one-time training cost."""
        
        total_tokens = num_examples * avg_tokens_per_example * epochs
        cost_per_1k = self.COSTS[model]['training']
        
        total_cost = (total_tokens / 1000) * cost_per_1k
        
        return total_cost
    
    def estimate_inference_savings(
        self,
        monthly_requests: int,
        tokens_saved_per_request: int,
        model: str = 'gpt-3.5-turbo'
    ) -> Dict[str, float]:
        """
        Estimate monthly savings from fine-tuning.
        
        Fine-tuning can reduce prompt tokens significantly
        by moving instructions into the model.
        """
        
        # Monthly token savings
        monthly_tokens_saved = monthly_requests * tokens_saved_per_request
        
        # Cost of those tokens at standard rate
        base_rate = 0.0010  # gpt-3.5-turbo input
        monthly_savings = (monthly_tokens_saved / 1000) * base_rate
        
        return {
            'monthly_savings_usd': monthly_savings,
            'annual_savings_usd': monthly_savings * 12,
            'tokens_saved_per_month': monthly_tokens_saved
        }
    
    def roi_analysis(
        self,
        training_examples: int,
        avg_tokens_per_example: int,
        epochs: int,
        monthly_requests: int,
        tokens_saved_per_request: int,
        model: str = 'gpt-3.5-turbo'
    ) -> Dict[str, Any]:
        """Calculate ROI for fine-tuning."""
        
        training_cost = self.estimate_training_cost(
            training_examples,
            avg_tokens_per_example,
            epochs,
            model
        )
        
        savings = self.estimate_inference_savings(
            monthly_requests,
            tokens_saved_per_request,
            model
        )
        
        # Months to break even
        breakeven_months = training_cost / savings['monthly_savings_usd']
        
        return {
            'one_time_training_cost': training_cost,
            'monthly_savings': savings['monthly_savings_usd'],
            'breakeven_months': breakeven_months,
            'worth_it': breakeven_months < 6,  # Worthwhile if ROI < 6 months
            'reasoning': f"Training costs \${training_cost: .2f
}, saves \${ savings['monthly_savings_usd']:.2f }/month"
        }

# Usage
calc = FineTuningCostCalculator()

roi = calc.roi_analysis(
    training_examples = 1000,
    avg_tokens_per_example = 200,
    epochs = 3,
    monthly_requests = 100000,
    tokens_saved_per_request = 150  # Moving system prompt into model
)

print(f"Training cost: \${roi['one_time_training_cost']:.2f}")
print(f"Monthly savings: \${roi['monthly_savings']:.2f}")
print(f"Break-even: {roi['breakeven_months']:.1f} months")

if roi['worth_it']:
    print("✅ Fine-tuning is worth it!")
else:
print("⚠️  ROI may not justify fine-tuning")
\`\`\`

## Best Practices

### 1. Data Quality Over Quantity

\`\`\`python
# ✅ GOOD: 500 high-quality, diverse examples
training_data = [
    {
        'input': 'Summarize this article about climate change...',
        'output': 'The article discusses...',
        'category': 'environment'
    },
    {
        'input': 'Summarize this tech news...',
        'output': 'The report highlights...',
        'category': 'technology'
    },
    # Diverse, high-quality examples across categories
]

# ❌ BAD: 5000 low-quality, repetitive examples
bad_training_data = [
    {'input': 'Summarize this', 'output': 'Summary'},
    {'input': 'Summarize that', 'output': 'Summary'},
    # Repetitive, low-quality
]
\`\`\`

### 2. Validation Set Monitoring

\`\`\`python
async def monitor_validation_loss(job_id: str):
    """Track validation loss during training."""
    
    events = await openai.FineTuningJob.list_events(job_id)
    
    train_losses = []
    val_losses = []
    
    for event in events['data']:
        if event['type'] == 'metrics':
            metrics = event['data']
            if 'train_loss' in metrics:
                train_losses.append(metrics['train_loss'])
            if 'valid_loss' in metrics:
                val_losses.append(metrics['valid_loss'])
    
    # Check for overfitting
    if val_losses:
        if val_losses[-1] > val_losses[0]:
            print("⚠️  Validation loss increasing - possible overfitting!")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
\`\`\`

### 3. Version Management

\`\`\`python
class ModelVersionManager:
    """Manage fine-tuned model versions."""
    
    def __init__(self):
        self.versions = {}
    
    def register_version(
        self,
        version: str,
        model_name: str,
        training_data_hash: str,
        metrics: Dict[str, float]
    ):
        """Register new model version."""
        self.versions[version] = {
            'model_name': model_name,
            'training_data_hash': training_data_hash,
            'metrics': metrics,
            'created_at': time.time(),
            'status': 'active'
        }
    
    def rollback(self, to_version: str):
        """Rollback to previous version."""
        if to_version in self.versions:
            # Set previous version as active
            for v in self.versions:
                self.versions[v]['status'] = 'inactive'
            self.versions[to_version]['status'] = 'active'
            
            print(f"✅ Rolled back to {to_version}")
            return self.versions[to_version]['model_name']
    
    def get_active_model(self) -> str:
        """Get currently active model."""
        for version, data in self.versions.items():
            if data['status'] == 'active':
                return data['model_name']

# Usage
manager = ModelVersionManager()

manager.register_version(
    version="v1.0",
    model_name="ft:gpt-3.5-turbo:org:custom:abc123",
    training_data_hash="hash_v1",
    metrics={'accuracy': 0.85}
)

manager.register_version(
    version="v1.1",
    model_name="ft:gpt-3.5-turbo:org:custom:xyz789",
    training_data_hash="hash_v2",
    metrics={'accuracy': 0.82}  # Worse!
)

# Rollback
manager.rollback("v1.0")
\`\`\`

## Troubleshooting

### Common Issues

**Issue 1: Training fails with "Invalid format"**
\`\`\`python
# Fix: Validate JSONL format
def validate_jsonl(file_path: str):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                assert 'messages' in data
                assert len(data['messages']) >= 2
            except Exception as e:
                print(f"❌ Line {i+1}: {e}")
                return False
    print("✅ JSONL valid")
    return True
\`\`\`

**Issue 2: Model not improving**
- Check data quality
- Increase epochs
- More diverse training examples
- Verify examples are correct

**Issue 3: Overfitting**
- Reduce epochs
- More validation data
- Regularization (built-in to OpenAI)

## Production Checklist

✅ **Pre-Training**
- [ ] 100+ high-quality examples
- [ ] Validation set prepared
- [ ] Data format validated
- [ ] Cost estimated and approved

✅ **Training**
- [ ] Job created successfully
- [ ] Monitoring validation loss
- [ ] Checkpoints tracked
- [ ] Training completes without errors

✅ **Post-Training**
- [ ] Model tested on held-out data
- [ ] Performance compared to base
- [ ] Version registered
- [ ] Rollback plan ready

✅ **Deployment**
- [ ] Gradual rollout strategy
- [ ] Monitoring in production
- [ ] Cost tracking enabled
- [ ] Feedback collection ready

## Next Steps

You now understand OpenAI fine-tuning. Next, learn:
- Fine-tuning open-source models
- RAG evaluation
- Continuous monitoring
- Building evaluation platforms
`,
};
