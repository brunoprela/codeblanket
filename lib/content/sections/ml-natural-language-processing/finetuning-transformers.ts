/**
 * Section: Fine-tuning Transformers
 * Module: Natural Language Processing
 *
 * Covers transfer learning, fine-tuning strategies, and adapting pre-trained models
 */

export const finetuningTransformersSection = {
  id: 'finetuning-transformers',
  title: 'Fine-tuning Transformers',
  content: `
# Fine-tuning Transformers

## Introduction

Fine-tuning adapts pre-trained transformers to specific tasks. Rather than training from scratch, we leverage knowledge from pre-training on massive corpora.

**Benefits:**
- **Transfer learning**: Leverage pre-trained knowledge
- **Data efficiency**: Works with small datasets
- **Faster training**: Starts from good weights
- **Better performance**: Often beats training from scratch

## Fine-tuning Workflow

\`\`\`python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# 1. Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 2. Prepare data
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# 4. Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# 5. Fine-tune
trainer.train()

# 6. Evaluate
results = trainer.evaluate()
print(results)
\`\`\`

## Fine-tuning Strategies

### Full Fine-tuning

\`\`\`python
# Update all parameters
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# All layers trainable
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
\`\`\`

### Frozen Layers (Feature Extraction)

\`\`\`python
# Freeze base model, train only classifier
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Freeze BERT layers
for param in model.bert.parameters():
    param.requires_grad = False

# Only train classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

# Much faster, less prone to overfitting
\`\`\`

### Gradual Unfreezing

\`\`\`python
# Train classifier first, then gradually unfreeze layers
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Phase 1: Train only classifier (1 epoch)
for param in model.bert.parameters():
    param.requires_grad = False
train(model, epochs=1)

# Phase 2: Unfreeze last encoder layer (1 epoch)
for param in model.bert.encoder.layer[-1].parameters():
    param.requires_grad = True
train(model, epochs=1)

# Phase 3: Unfreeze all (1 epoch)
for param in model.parameters():
    param.requires_grad = True
train(model, epochs=1)
\`\`\`

### Discriminative Fine-tuning (Different LRs per Layer)

\`\`\`python
# Lower layers: smaller LR (general features)
# Higher layers: larger LR (task-specific)

optimizer_grouped_parameters = [
    {'params': model.bert.embeddings.parameters(), 'lr': 1e-5},
    {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 2e-5},
    {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
\`\`\`

## Parameter-Efficient Fine-tuning

### LoRA (Low-Rank Adaptation)

\`\`\`python
from peft import LoraConfig, get_peft_model

# Only train small number of parameters
config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Which layers to adapt
)

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
model = get_peft_model(model, config)

# Trainable parameters: ~0.1% of total!
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,483,778 || trainable%: 0.27%
\`\`\`

### Adapter Layers

\`\`\`python
# Add small adapter modules between transformer layers
class AdapterLayer(nn.Module):
    def __init__(self, d_model, adapter_size=64):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_size)
        self.up = nn.Linear(adapter_size, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))

# Freeze base model, train only adapters
\`\`\`

### Prefix Tuning

\`\`\`python
# Add trainable prefix tokens to input
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=20
)

model = get_peft_model(model, config)
\`\`\`

## Best Practices

### 1. Learning Rate Selection

\`\`\`python
# Recommended learning rates:
lr_recommendations = {
    'base_model': 2e-5,  # BERT layers
    'classifier': 1e-4,  # Task-specific head
    'adapter': 1e-3,     # Adapter layers
}

# Learning rate warmup
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,  # Warm up 10% of steps
    num_training_steps=5000
)
\`\`\`

### 2. Batch Size and Gradient Accumulation

\`\`\`python
# Small GPU memory? Use gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Small batch fits in memory
    gradient_accumulation_steps=4,  # Effective batch size = 4*4 = 16
    fp16=True,  # Mixed precision training
)
\`\`\`

### 3. Regularization

\`\`\`python
training_args = TrainingArguments(
    weight_decay=0.01,  # L2 regularization
    dropout=0.1,        # Dropout rate
    attention_probs_dropout_prob=0.1,
    hidden_dropout_prob=0.1,
)

# Early stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
\`\`\`

### 4. Data Augmentation

\`\`\`python
# Back-translation
def augment_text(text, model='en-de-en'):
    # Translate to German and back to English
    de_text = translate(text, 'en', 'de')
    back_text = translate(de_text, 'de', 'en')
    return back_text

# Synonym replacement
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet')
augmented = aug.augment(text)

# Random deletion/swap
# Increase effective training data
\`\`\`

## Task-Specific Fine-tuning

### Sentiment Classification

\`\`\`python
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # Positive, Negative, Neutral
)

# Dataset format
dataset = {
    'text': ['I love this!', 'Terrible product'],
    'label': [2, 0]  # 0=negative, 1=neutral, 2=positive
}
\`\`\`

### Named Entity Recognition

\`\`\`python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=9,  # B-PER, I-PER, B-LOC, I-LOC, etc.
)
\`\`\`

### Question Answering

\`\`\`python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# Dataset format
context = "Paris is the capital of France"
question = "What is the capital of France?"
# Predict start and end positions of answer span
\`\`\`

## Monitoring and Debugging

\`\`\`python
# Track metrics with Weights & Biases
import wandb
wandb.init(project="fine-tuning")

training_args = TrainingArguments(
    report_to="wandb",
    logging_steps=10,
)

# Log custom metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    from sklearn.metrics import accuracy_score, f1_score
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)
\`\`\`

## Common Pitfalls

\`\`\`python
# 1. Learning rate too high
# Symptom: Loss explodes, NaN
# Fix: Use 2e-5 for BERT, warmup

# 2. Overfitting
# Symptom: Train accuracy >> val accuracy
# Fix: More regularization, early stopping

# 3. Catastrophic forgetting
# Symptom: Poor performance on general tasks after fine-tuning
# Fix: Lower learning rate, freeze lower layers

# 4. Out of memory
# Symptom: CUDA OOM error
# Fix: Smaller batch size, gradient accumulation, fp16

# 5. Wrong tokenizer
# Symptom: Poor performance, weird predictions
# Fix: Use same tokenizer as pre-trained model
\`\`\`

## Summary

Fine-tuning transformers:
- **Transfer learning**: Leverage pre-trained knowledge
- **Strategies**: Full, frozen, gradual, discriminative
- **Efficient**: LoRA, adapters reduce trainable parameters
- **Best practices**: Proper LR, warmup, regularization
- **Task-specific**: Different heads for different tasks

**Next**: Applying fine-tuned models to text classification.
`,
};
