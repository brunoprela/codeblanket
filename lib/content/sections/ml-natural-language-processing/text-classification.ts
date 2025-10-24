/**
 * Section: Text Classification
 * Module: Natural Language Processing
 *
 * Comprehensive coverage of sentiment analysis, topic classification, multi-class and multi-label text classification
 */

export const textClassificationSection = {
  id: 'text-classification',
  title: 'Text Classification',
  content: `
# Text Classification

## Introduction

Text classification is the task of assigning predefined categories or labels to text documents. It's one of the most fundamental and widely-used NLP tasks with applications spanning sentiment analysis, spam detection, topic categorization, intent classification, and content moderation.

**Why Text Classification Matters:**

- **Sentiment Analysis**: Understand customer opinions and emotions
- **Content Moderation**: Filter harmful or inappropriate content
- **Topic Categorization**: Organize documents by subject
- **Intent Recognition**: Understand user intentions in chatbots
- **Spam Detection**: Filter unwanted messages
- **Document Routing**: Automatically route documents to correct departments
- **News Classification**: Categorize news articles by topic
- **Financial Analysis**: Classify financial news as bullish/bearish

**Types of Text Classification:**

\`\`\`
Binary Classification:
- Spam vs Not Spam
- Positive vs Negative sentiment
- Relevant vs Irrelevant

Multi-Class Classification (single label):
- News categories: Sports, Politics, Technology, Entertainment
- Customer support: Billing, Technical, Sales, General

Multi-Label Classification (multiple labels):
- Movie genres: Action, Comedy, Drama (can have multiple)
- Research paper topics: ML, NLP, CV (can have multiple)

Hierarchical Classification:
- News → Politics → International → Middle East
\`\`\`

## Sentiment Analysis

Sentiment analysis (opinion mining) determines the emotional tone of text: positive, negative, or neutral.

### Binary Sentiment Classification

\`\`\`python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Method 1: Using pre-trained pipeline (easiest)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I absolutely love this product! Best purchase ever!",
    "Terrible experience. Would not recommend.",
    "It's okay, nothing special but not bad either.",
]

results = sentiment_pipeline(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.3f})")
    print("-" * 60)

# Output:
# Text: I absolutely love this product! Best purchase ever!
# Sentiment: POSITIVE (confidence: 0.999)
# ------------------------------------------------------------
# Text: Terrible experience. Would not recommend.
# Sentiment: NEGATIVE (confidence: 0.998)
# ------------------------------------------------------------
# Text: It's okay, nothing special but not bad either.
# Sentiment: POSITIVE (confidence: 0.682)  # Less confident (neutral text)

# Method 2: Manual approach (more control)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_sentiment(text, threshold=0.9):
    """
    Predict sentiment with confidence thresholding
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence = probs.max().item()
    predicted_class = probs.argmax().item()
    
    # Map to label
    label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    
    # Handle uncertain predictions
    if confidence < threshold:
        label = "NEUTRAL (uncertain)"
    
    return {
        'label': label,
        'confidence': confidence,
        'probabilities': {
            'negative': probs[0][0].item(),
            'positive': probs[0][1].item()
        }
    }

# Test
text = "This product is okay, not great but not terrible."
result = predict_sentiment(text, threshold=0.8)
print(f"\\nText: {text}")
print(f"Prediction: {result}")
\`\`\`

### Multi-Class Sentiment Analysis

\`\`\`python
# 5-class sentiment: Very Negative, Negative, Neutral, Positive, Very Positive

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# Load SST-5 dataset (5-class sentiment)
# You can also use custom data

# Example custom dataset
from datasets import Dataset

data = {
    'text': [
        "This is absolutely amazing! Best thing ever!",
        "Pretty good, I like it",
        "It's okay, nothing special",
        "Not great, disappointed",
        "Terrible! Complete waste of money!",
    ],
    'label': [4, 3, 2, 1, 0]  # 0: Very Negative, 4: Very Positive
}

dataset = Dataset.from_dict(data)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=5,
    id2label={0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'},
    label2id={'Very Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Very Positive': 4}
)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# For full training, you would split into train/val/test and train the model
# Here we'll just show prediction

# Predict
def predict_multi_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = probs.argmax().item()
    
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    
    return {
        'label': labels[predicted_class],
        'confidence': probs.max().item(),
        'all_probabilities': {label: prob.item() for label, prob in zip(labels, probs[0])}
    }

# Test
text = "The product works but could be better"
result = predict_multi_sentiment(text)
print(f"\\nText: {text}")
print(f"Predicted sentiment: {result['label']} ({result['confidence']:.3f})")
print("\\nAll probabilities:")
for label, prob in result['all_probabilities'].items():
    print(f"  {label}: {prob:.3f}")
\`\`\`

### Aspect-Based Sentiment Analysis

Analyze sentiment towards specific aspects of a product/service:

\`\`\`python
# Example: Restaurant review
review = """
The food was absolutely delicious and beautifully presented. 
However, the service was terrible - we waited 45 minutes for our order. 
The ambiance was nice and cozy. Prices are a bit high but reasonable 
for the quality of food.
"""

# Aspects: Food, Service, Ambiance, Price
# For each aspect, extract sentiment

aspects = {
    'Food': ['food', 'delicious', 'quality'],
    'Service': ['service', 'waited'],
    'Ambiance': ['ambiance', 'cozy'],
    'Price': ['price', 'high', 'reasonable']
}

def aspect_sentiment(text, aspects):
    """Simple aspect-based sentiment analysis"""
    results = {}
    
    # Split into sentences
    sentences = text.split('.')
    
    for aspect, keywords in aspects.items():
        aspect_sentences = []
        
        # Find sentences mentioning the aspect
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                aspect_sentences.append(sentence.strip())
        
        # Analyze sentiment of aspect sentences
        if aspect_sentences:
            aspect_text = '. '.join(aspect_sentences)
            sentiment_result = sentiment_pipeline(aspect_text)[0]
            results[aspect] = {
                'sentiment': sentiment_result['label'],
                'confidence': sentiment_result['score'],
                'context': aspect_text
            }
    
    return results

aspect_results = aspect_sentiment(review, aspects)

print("Aspect-Based Sentiment Analysis:")
print("=" * 60)
for aspect, result in aspect_results.items():
    print(f"\\n{aspect}:")
    print(f"  Sentiment: {result['sentiment']} ({result['confidence']:.3f})")
    print(f"  Context: {result['context']}")
\`\`\`

## Topic Classification

Categorizing documents by topic or subject matter:

\`\`\`python
from transformers import pipeline
from datasets import load_dataset

# Load AG News dataset (4 classes: World, Sports, Business, Sci/Tech)
dataset = load_dataset('ag_news')

print("AG News Categories:")
print(dataset['train'].features['label'].names)
# ['World', 'Sports', 'Business', 'Sci/Tech']

# Sample data
print("\\nSample:")
sample = dataset['train'][0]
print(f"Text: {sample['text']}")
print(f"Label: {dataset['train'].features['label'].names[sample['label']]}")

# Train topic classifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

tokenized_train = dataset['train'].map(tokenize_function, batched=True)
tokenized_test = dataset['test'].map(tokenize_function, batched=True)

# Load model
num_labels = 4
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./topic_classifier',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train (commented out - takes time)
# trainer.train()

# Save model
# trainer.save_model('./topic_classifier_final')

# Predict on new text
def predict_topic(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = probs.argmax().item()
    
    topics = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    return {
        'topic': topics[predicted_class],
        'confidence': probs.max().item(),
        'all_probabilities': {topic: prob.item() for topic, prob in zip(topics, probs[0])}
    }

# Test
text = "Apple Inc. announced record quarterly earnings with revenue exceeding $100 billion."
result = predict_topic(text)
print(f"\\nText: {text}")
print(f"Topic: {result['topic']} ({result['confidence']:.3f})")
\`\`\`

## Multi-Label Classification

Documents can have multiple labels simultaneously:

\`\`\`python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, hamming_loss

# Example: Movie genre classification (multiple genres per movie)
data = {
    'text': [
        "An action-packed adventure with thrilling stunts and romance",
        "A hilarious comedy about friendship and life lessons",
        "Intense crime drama with complex characters and twists",
    ],
    'labels': [
        [1, 1, 0, 1, 0],  # Action, Adventure, Comedy, Romance, Drama
        [0, 0, 1, 0, 0],  # Comedy
        [0, 0, 0, 0, 1],  # Drama
    ]
}

genres = ['Action', 'Adventure', 'Comedy', 'Romance', 'Drama']

# Multi-label classifier
class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Sigmoid for multi-label (independent probabilities)
        return torch.sigmoid(logits)

# Initialize model
model = MultiLabelClassifier('bert-base-uncased', num_labels=len(genres))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Training loop (simplified)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCELoss()  # Binary cross-entropy for multi-label

# Prepare data
texts = data['text']
labels = torch.tensor(data['labels'], dtype=torch.float32)

# Training step (one batch example)
model.train()

# Tokenize
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# Forward pass
optimizer.zero_grad()
outputs = model(encoded['input_ids'], encoded['attention_mask'])
loss = criterion(outputs, labels)

# Backward pass
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")

# Prediction
model.eval()

def predict_genres(text, threshold=0.5):
    """Predict multiple genres"""
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        probs = model(encoded['input_ids'], encoded['attention_mask'])
    
    # Apply threshold
    predictions = (probs[0] > threshold).int().tolist()
    
    # Get predicted genres
    predicted_genres = [genre for genre, pred in zip(genres, predictions) if pred == 1]
    
    # Get all probabilities
    all_probs = {genre: prob.item() for genre, prob in zip(genres, probs[0])}
    
    return {
        'genres': predicted_genres,
        'probabilities': all_probs
    }

# Test
text = "A thrilling action movie with amazing special effects and a touching love story"
result = predict_genres(text, threshold=0.4)

print(f"\\nText: {text}")
print(f"\\nPredicted Genres: {', '.join(result['genres'])}")
print("\\nAll Probabilities:")
for genre, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {genre}: {prob:.3f}")
\`\`\`

## Fine-Tuning for Custom Classification

Complete pipeline for training a custom classifier:

\`\`\`python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

# Example: Customer support ticket classification
tickets = pd.DataFrame({
    'text': [
        "My account is locked and I can't log in",
        "I want to cancel my subscription immediately",
        "How do I reset my password?",
        "The app keeps crashing on my iPhone",
        "Can I upgrade to the premium plan?",
        "I was charged twice for the same purchase",
        "What features are included in the basic plan?",
        "The website won't load on Chrome",
    ],
    'category': [
        'Account', 'Billing', 'Account', 'Technical',
        'Sales', 'Billing', 'Sales', 'Technical'
    ]
})

# Encode labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
tickets['label'] = label_encoder.fit_transform(tickets['category'])

print("Label Mapping:")
for i, category in enumerate(label_encoder.classes_):
    print(f"  {i}: {category}")

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    tickets['text'].tolist(),
    tickets['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=tickets['label']
)

# Create datasets
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

test_dataset = Dataset.from_dict({
    'text': test_texts,
    'label': test_labels
})

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Load model and tokenizer
model_name = 'distilbert-base-uncased'
num_labels = len(label_encoder.classes_)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(label_encoder.classes_)},
    label2id={label: i for i, label in enumerate(label_encoder.classes_)}
)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./ticket_classifier',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    warmup_steps=100,
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
print("Starting training...")
trainer.train()

# Evaluate
print("\\nEvaluating on test set...")
results = trainer.evaluate()

print("\\nTest Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")

# Save model
trainer.save_model('./ticket_classifier_final')
tokenizer.save_pretrained('./ticket_classifier_final')

print("\\nModel saved!")

# Inference function
def classify_ticket(text):
    """Classify customer support ticket"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = probs.argmax().item()
    
    return {
        'category': label_encoder.classes_[predicted_class],
        'confidence': probs.max().item(),
        'all_probabilities': {
            label_encoder.classes_[i]: prob.item()
            for i, prob in enumerate(probs[0])
        }
    }

# Test on new ticket
new_ticket = "I need help setting up automatic payments"
result = classify_ticket(new_ticket)

print(f"\\nNew Ticket: {new_ticket}")
print(f"Category: {result['category']} (confidence: {result['confidence']:.3f})")
print("\\nAll probabilities:")
for category, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {category}: {prob:.3f}")
\`\`\`

## Evaluation Metrics for Text Classification

\`\`\`python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Example predictions
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
y_pred = [0, 2, 2, 0, 1, 2, 1, 1, 2, 0]

categories = ['Sports', 'Politics', 'Technology']

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Overall Metrics:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1 Score:  {f1:.3f}")

# Per-class metrics
print("\\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=categories))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=categories,
    yticklabels=categories
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# For binary classification: ROC curve
# (Example with binary probabilities)
y_true_binary = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1]
y_probs = [0.1, 0.9, 0.8, 0.2, 0.85, 0.3, 0.75, 0.15, 0.9, 0.7]

fpr, tpr, thresholds = roc_curve(y_true_binary, y_probs)
roc_auc = roc_auc_score(y_true_binary, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()
\`\`\`

## Handling Class Imbalance

\`\`\`python
from sklearn.utils.class_weight import compute_class_weight
import torch

# Imbalanced dataset example
y_train = [0]*100 + [1]*20 + [2]*5  # Severely imbalanced

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

print("Class weights:", class_weights)

# Method 1: Weighted loss
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Method 2: Oversampling (imblearn)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# For text, we need to vectorize first
from sklearn.feature_extraction.text import TfidfVectorizer

# Example texts
texts_imbalanced = ["text"] * 100 + ["text"] * 20 + ["text"] * 5
labels_imbalanced = [0]*100 + [1]*20 + [2]*5

# Vectorize
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(texts_imbalanced)

# SMOTE for oversampling minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, labels_imbalanced)

print(f"\\nOriginal distribution: {np.bincount(labels_imbalanced)}")
print(f"Resampled distribution: {np.bincount(y_resampled)}")

# Method 3: Stratified sampling
# Ensure balanced splits during train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts_imbalanced,
    labels_imbalanced,
    test_size=0.2,
    stratify=labels_imbalanced,  # Maintain class proportions
    random_state=42
)
\`\`\`

## Best Practices

### 1. Data Quality

\`\`\`python
# Check for data quality issues

def analyze_dataset(texts, labels):
    """Analyze text classification dataset"""
    # Label distribution
    from collections import Counter
    label_dist = Counter(labels)
    
    print("Label Distribution:")
    for label, count in label_dist.items():
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Text length statistics
    text_lengths = [len(text.split()) for text in texts]
    
    print(f"\\nText Length Statistics:")
    print(f"  Mean: {np.mean(text_lengths):.1f} words")
    print(f"  Median: {np.median(text_lengths):.1f} words")
    print(f"  Min: {min(text_lengths)} words")
    print(f"  Max: {max(text_lengths)} words")
    
    # Check for duplicates
    unique_texts = len(set(texts))
    duplicates = len(texts) - unique_texts
    
    print(f"\\nDuplicate texts: {duplicates}")
    
    # Check for empty/very short texts
    short_texts = sum(1 for text in texts if len(text.split()) < 3)
    print(f"Very short texts (<3 words): {short_texts}")

analyze_dataset(tickets['text'].tolist(), tickets['category'].tolist())
\`\`\`

### 2. Model Selection

\`\`\`
Small Dataset (<10K samples):
- Use pre-trained models and fine-tune
- Start with smaller models (DistilBERT, ALBERT)
- Consider data augmentation

Medium Dataset (10K-100K samples):
- Fine-tune BERT-base or RoBERTa
- Experiment with learning rates and batch sizes

Large Dataset (>100K samples):
- Can train larger models (BERT-large, RoBERTa-large)
- Consider training from scratch for domain-specific tasks

Real-time Requirements:
- Use smaller/faster models (DistilBERT, MobileBERT)
- Quantization and optimization
- Cache frequent predictions
\`\`\`

### 3. Hyperparameter Tuning

\`\`\`python
from transformers import Trainer, TrainingArguments
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

def hyperparameter_search():
    """Hyperparameter search with Ray Tune"""
    
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
    
    # Define search space
    search_space = {
        'learning_rate': tune.loguniform(1e-5, 5e-5),
        'per_device_train_batch_size': tune.choice([8, 16, 32]),
        'num_train_epochs': tune.choice([3, 4, 5]),
        'weight_decay': tune.uniform(0.0, 0.3),
    }
    
    # Training args
    training_args = TrainingArguments(
        output_dir='./hyperparam_search',
        evaluation_strategy='epoch',
        save_strategy='epoch',
    )
    
    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )
    
    # Run hyperparameter search
    best_trial = trainer.hyperparameter_search(
        hp_space=lambda _: search_space,
        backend='ray',
        n_trials=10,
        direction='maximize',
        compute_objective=lambda metrics: metrics['eval_f1']
    )
    
    return best_trial

# Run search (commented out - takes time)
# best = hyperparameter_search()
# print(f"Best hyperparameters: {best.hyperparameters}")
\`\`\`

## Financial Text Classification Application

Classify financial news sentiment for trading signals:

\`\`\`python
# Financial news sentiment classifier
financial_texts = [
    "Apple reports record Q1 earnings, stock surges 5%",
    "Tesla faces regulatory challenges in China, shares drop",
    "Fed signals potential rate hike amid inflation concerns",
    "Tech sector rallies on strong employment data",
]

# Use financial-specific model
financial_sentiment = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"  # FinBERT: financial sentiment model
)

results = financial_sentiment(financial_texts)

print("Financial News Sentiment Analysis:")
print("=" * 60)
for text, result in zip(financial_texts, results):
    print(f"\\nNews: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.3f})")
    
    # Trading signal (simplified)
    if result['label'] == 'positive' and result['score'] > 0.9:
        signal = "STRONG BUY"
    elif result['label'] == 'positive':
        signal = "BUY"
    elif result['label'] == 'negative' and result['score'] > 0.9:
        signal = "STRONG SELL"
    elif result['label'] == 'negative':
        signal = "SELL"
    else:
        signal = "HOLD"
    
    print(f"Trading Signal: {signal}")
\`\`\`

## Summary

Text Classification:
- **Versatile**: Sentiment, topic, intent, spam detection, etc.
- **Transformer-based**: BERT, RoBERTa, DistilBERT achieve SOTA
- **Multi-class**: Single label from multiple categories
- **Multi-label**: Multiple labels per document (e.g., genres)
- **Fine-tuning**: Pre-trained models + domain data = excellent results
- **Class Imbalance**: Use class weights, resampling, stratified splits
- **Evaluation**: Accuracy, precision, recall, F1, confusion matrix
- **Production**: Fast inference with DistilBERT, caching, batching
- **Financial NLP**: Specialized models (FinBERT) for financial sentiment

**Next**: Apply classification in Named Entity Recognition and Question Answering tasks.
`,
};
