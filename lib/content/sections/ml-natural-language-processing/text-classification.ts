/**
 * Section: Text Classification
 * Module: Natural Language Processing
 *
 * Covers sentiment analysis, topic classification, and text categorization
 */

export const textClassificationSection = {
  id: 'text-classification',
  title: 'Text Classification',
  content: `
# Text Classification

## Introduction

Text classification assigns labels to text documents. It's one of the most common NLP tasks with applications in sentiment analysis, spam detection, topic categorization, and more.

## Sentiment Analysis with Transformers

\`\`\`python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Using pre-trained model
classifier = pipeline('sentiment-analysis')
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]

# Custom fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions
\`\`\`

## Multi-class Classification

\`\`\`python
# Topic classification example
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Load AG News dataset (4 classes: World, Sports, Business, Technology)
dataset = load_dataset('ag_news')

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Train
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()
\`\`\`

## Multi-label Classification

\`\`\`python
# Document can have multiple labels
from sklearn.metrics import f1_score
import torch.nn as nn

class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return torch.sigmoid(logits)  # Sigmoid for multi-label

# Training with BCELoss
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch['input_ids'], batch['attention_mask'])
    loss = criterion(logits, batch['labels'].float())
    loss.backward()
    optimizer.step()
\`\`\`

## Evaluation Metrics

\`\`\`python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# Classification report
print(classification_report(labels, preds, target_names=['Class0', 'Class1', 'Class2']))

# Confusion matrix
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# F1 scores
from sklearn.metrics import f1_score
f1_micro = f1_score(labels, preds, average='micro')
f1_macro = f1_score(labels, preds, average='macro')
f1_weighted = f1_score(labels, preds, average='weighted')
\`\`\`

## Best Practices

1. **Class Imbalance**: Use weighted loss
2. **Evaluation**: Use F1 for imbalanced data
3. **Regularization**: Dropout, early stopping
4. **Ensemble**: Combine multiple models

## Summary

Text classification with transformers achieves state-of-the-art results across tasks.
`,
};
