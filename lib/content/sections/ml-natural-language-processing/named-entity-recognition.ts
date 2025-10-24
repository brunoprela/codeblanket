/**
 * Section: Named Entity Recognition
 * Module: Natural Language Processing
 *
 * Comprehensive coverage of NER, token classification, entity extraction, and production NER systems
 */

export const namedEntityRecognitionSection = {
  id: 'named-entity-recognition',
  title: 'Named Entity Recognition',
  content: `
# Named Entity Recognition (NER)

## Introduction

Named Entity Recognition (NER) is a fundamental NLP task that identifies and classifies named entities in text. Entities include people, organizations, locations, dates, monetary values, percentages, and domain-specific entities.

**Why NER Matters:**

- **Information Extraction**: Extract structured data from unstructured text
- **Knowledge Graph Construction**: Build entity relationships
- **Search & Recommendation**: Improve search by understanding entities
- **Document Understanding**: Identify key actors and subjects
- **Financial Analysis**: Extract companies, people, and financial metrics
- **Question Answering**: Identify entities relevant to questions

**Real-World Applications:**

\`\`\`
Financial News: "Apple Inc. announced a $50 billion buyback program"
Entities:
- Apple Inc. → Organization
- $50 billion → Monetary Value
- buyback program → Financial Instrument

Medical Record: "Dr. Smith prescribed 500mg of Aspirin to John Doe"
Entities:
- Dr. Smith → Person (Medical Professional)
- 500mg → Dosage
- Aspirin → Drug
- John Doe → Person (Patient)

Legal Document: "The case was filed in the Southern District of New York on March 15, 2024"
Entities:
- Southern District of New York → Location (Court)
- March 15, 2024 → Date
\`\`\`

**NER as Sequence Labeling:**

NER is typically framed as a token classification problem where each token receives a label indicating its entity type (or non-entity status).

## IOB/BIO Tagging Scheme

The IOB (Inside-Outside-Beginning) tagging scheme is the standard for representing entities at the token level:

\`\`\`
B-ENTITY: Beginning of an entity
I-ENTITY: Inside/continuation of an entity
O: Outside any entity (not an entity)

Example: "Apple Inc. was founded by Steve Jobs in California"

Tokens:  Apple  Inc.  was  founded  by  Steve  Jobs  in  California
Labels:  B-ORG  I-ORG  O    O        O   B-PER  I-PER  O   B-LOC

Explanation:
- "Apple Inc." is one organization entity (B-ORG, I-ORG)
- "Steve Jobs" is one person entity (B-PER, I-PER)
- "California" is a location entity (B-LOC)
- Other tokens are not entities (O)
\`\`\`

**Why B- and I- Tags Matter:**

\`\`\`
Without distinction (just ORG, PER, LOC):
"Apple Microsoft" → ORG ORG (ambiguous - one entity or two?)

With IOB:
"Apple Microsoft" → B-ORG B-ORG (clearly two separate entities)
"Apple Inc." → B-ORG I-ORG (clearly one entity)
\`\`\`

**BILOU Tagging (Extended):**

Some systems use more detailed tagging:

\`\`\`
B-ENTITY: Beginning
I-ENTITY: Inside
L-ENTITY: Last token
U-ENTITY: Unit (single-token entity)
O: Outside

Example: "Apple hired John"
Tokens: Apple hired John
BILOU:  U-ORG  O    U-PER (both are single-token entities)

Example: "Apple Inc. hired Steve Jobs"
Tokens: Apple  Inc.  hired  Steve  Jobs
BILOU:  B-ORG  L-ORG  O     B-PER  L-PER
\`\`\`

## Common Entity Types

### Standard Entity Types (CoNLL-2003):

\`\`\`
PER (Person): Names of people
- "Barack Obama", "Marie Curie", "Dr. Smith"

ORG (Organization): Companies, agencies, institutions
- "Apple Inc.", "United Nations", "Harvard University"

LOC (Location): Countries, cities, geographic features
- "New York", "Mount Everest", "Pacific Ocean"

MISC (Miscellaneous): Other named entities
- "Olympics", "Nobel Prize", "iPhone"
\`\`\`

### Extended Entity Types:

\`\`\`
DATE: Dates and date ranges
- "March 15, 2024", "Q1 2023", "yesterday"

TIME: Times
- "3:00 PM", "morning", "midnight"

MONEY: Monetary values
- "$1 million", "€50", "100 dollars"

PERCENT: Percentages
- "50%", "three percent", "25 basis points"

GPE (Geo-Political Entity): Countries, states, cities with government
- "United States", "California", "Paris"

PRODUCT: Products and services
- "iPhone 15", "Windows 11", "ChatGPT"

EVENT: Named events
- "World War II", "Olympics 2024", "Super Bowl"

LAW: Laws and regulations
- "GDPR", "Clean Air Act", "Dodd-Frank"
\`\`\`

### Domain-Specific Entity Types:

**Financial Entities:**
\`\`\`
TICKER: Stock ticker symbols (AAPL, TSLA)
CUSIP: Security identifiers
EXCHANGE: Stock exchanges (NYSE, NASDAQ)
INSTRUMENT: Financial instruments (bonds, options)
\`\`\`

**Medical Entities:**
\`\`\`
DISEASE: Medical conditions (diabetes, COVID-19)
DRUG: Medications (Aspirin, Metformin)
SYMPTOM: Symptoms (fever, cough)
TEST: Medical tests (MRI, blood test)
\`\`\`

**Legal Entities:**
\`\`\`
COURT: Courts (Supreme Court, District Court)
STATUTE: Laws and statutes (18 USC 242)
JUDGE: Judge names
ATTORNEY: Attorney names
\`\`\`

## NER with spaCy

spaCy provides fast, production-ready NER:

\`\`\`python
import spacy
from spacy import displacy

# Load pre-trained model (with NER)
nlp = spacy.load("en_core_web_sm")

text = """
Apple Inc. announced its Q1 earnings on January 28, 2024. 
CEO Tim Cook reported revenue of $119.6 billion, exceeding 
analyst expectations. The company's stock (AAPL) rose 5% to $180.
"""

# Process text
doc = nlp(text)

# Extract entities
print("\\nEntities found:")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:20} | {ent.label_:10} | {ent.start_char:3}-{ent.end_char:3}")

# Output:
# Apple Inc.           | ORG        |   1-11
# Q1                   | DATE       |  28-30
# January 28, 2024     | DATE       |  43-59
# Tim Cook             | PERSON     |  66-74
# $119.6 billion       | MONEY      |  95-109
# AAPL                 | ORG        | 159-163
# 5%                   | PERCENT    | 170-172
# $180                 | MONEY      | 176-180

# Visualize entities (in Jupyter notebook)
displacy.render(doc, style="ent", jupyter=True)

# Get entity statistics
from collections import Counter

entity_counts = Counter([ent.label_ for ent in doc.ents])
print("\\nEntity Distribution:")
for entity_type, count in entity_counts.most_common():
    print(f"{entity_type}: {count}")
\`\`\`

**Customizing Entity Recognition:**

\`\`\`python
# Add custom entity patterns
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# Pattern for stock tickers (1-5 uppercase letters)
ticker_pattern = [{"TEXT": {"REGEX": "^[A-Z]{1,5}$"}}]
matcher.add("TICKER", [ticker_pattern])

doc = nlp("Invested in AAPL, TSLA, and MSFT")
matches = matcher(doc)

for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Ticker found: {span.text}")
\`\`\`

## NER with Transformers

Transformer-based NER achieves state-of-the-art performance:

\`\`\`python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load pre-trained NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"  # Merge subword tokens
)

text = "Apple Inc. CEO Tim Cook announced new products in Cupertino, California."

# Predict entities
entities = ner_pipeline(text)

print("Entities:")
for entity in entities:
    print(f"{entity['word']:20} | {entity['entity_group']:10} | Score: {entity['score']:.3f}")

# Output:
# Apple Inc.           | ORG        | Score: 0.997
# Tim Cook             | PER        | Score: 0.999
# Cupertino            | LOC        | Score: 0.998
# California           | LOC        | Score: 0.999
\`\`\`

**Handling Aggregation Strategies:**

\`\`\`python
# Different aggregation strategies

# 1. No aggregation (raw subwords)
entities_raw = ner_pipeline(text, aggregation_strategy="none")
for ent in entities_raw[:3]:
    print(f"{ent['word']:15} | {ent['entity']:15} | {ent['score']:.3f}")

# Output:
# Apple          | B-ORG          | 0.999
# Inc            | I-ORG          | 0.998
# .              | I-ORG          | 0.912

# 2. Simple aggregation (merge B- and I- tags)
entities_simple = ner_pipeline(text, aggregation_strategy="simple")
print(entities_simple[0])
# {'entity_group': 'ORG', 'score': 0.997, 'word': 'Apple Inc.', 'start': 0, 'end': 10}

# 3. First token score
entities_first = ner_pipeline(text, aggregation_strategy="first")

# 4. Average score
entities_avg = ner_pipeline(text, aggregation_strategy="average")

# 5. Maximum score
entities_max = ner_pipeline(text, aggregation_strategy="max")
\`\`\`

## Fine-Tuning NER Models

Train custom NER models for domain-specific entities:

\`\`\`python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score

# 1. Load and prepare data
dataset = load_dataset("conll2003")

# Label list: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
label_list = dataset['train'].features['ner_tags'].feature.names
num_labels = len(label_list)

print(f"Labels: {label_list}")
print(f"Number of labels: {num_labels}")

# 2. Load model and tokenizer
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
)

# 3. Tokenize and align labels
def tokenize_and_align_labels(examples):
    """
    Tokenize text and align NER labels with subword tokens.
    
    Challenge: BERT tokenizer may split words into subwords
    Example: "Apple" → ["App", "##le"]
    We need to assign labels to each subword.
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get label -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the actual label
                label_ids.append(label[word_idx])
            else:
                # Subsequent subword tokens get -100 (ignored in loss)
                # Alternative: could use I- tag
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Tokenize datasets
tokenized_train = dataset['train'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['train'].column_names
)

tokenized_val = dataset['validation'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['validation'].column_names
)

tokenized_test = dataset['test'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['test'].column_names
)

# 4. Data collator (handles padding and batching)
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)

# 5. Evaluation metrics
def compute_metrics(eval_preds):
    """
    Compute NER metrics using seqeval (entity-level metrics)
    """
    predictions, labels = eval_preds
    
    # Get predicted labels (argmax of logits)
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (-100) and convert to label names
    true_labels = []
    pred_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_label = []
        pred_label = []
        
        for pred, lab in zip(prediction, label):
            if lab != -100:  # Ignore special tokens
                true_label.append(label_list[lab])
                pred_label.append(label_list[pred])
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
    
    # Compute entity-level metrics (seqeval)
    results = {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
    }
    
    return results

# Import seqeval metrics
from seqeval.metrics import precision_score, recall_score, f1_score

# 6. Training arguments
training_args = TrainingArguments(
    output_dir='./ner_model',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    warmup_steps=500,
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train
print("Starting training...")
trainer.train()

# 9. Evaluate on test set
print("\\nEvaluating on test set...")
test_results = trainer.predict(tokenized_test)

# Get predictions
predictions = np.argmax(test_results.predictions, axis=2)

# Convert to labels
true_labels = []
pred_labels = []

for prediction, label in zip(predictions, test_results.label_ids):
    true_label = []
    pred_label = []
    
    for pred, lab in zip(prediction, label):
        if lab != -100:
            true_label.append(label_list[lab])
            pred_label.append(label_list[pred])
    
    true_labels.append(true_label)
    pred_labels.append(pred_label)

# Detailed classification report
print("\\nDetailed Results:")
print(classification_report(true_labels, pred_labels))

# Save model
trainer.save_model('./ner_model_final')
tokenizer.save_pretrained('./ner_model_final')

print("\\nModel saved to ./ner_model_final")
\`\`\`

## Custom Domain NER

Building NER for financial entities:

\`\`\`python
# Example: Financial NER dataset
financial_data = [
    {
        "text": "Apple (AAPL) reported Q1 earnings of $50B",
        "entities": [
            (0, 5, "COMPANY"),
            (7, 11, "TICKER"),
            (22, 24, "QUARTER"),
            (36, 40, "MONEY")
        ]
    },
    {
        "text": "Tesla stock rose 10% after Elon Musk's announcement",
        "entities": [
            (0, 5, "COMPANY"),
            (16, 19, "PERCENT"),
            (26, 35, "PERSON")
        ]
    },
    # ... more examples
]

# Convert to CoNLL format
def convert_to_conll(data):
    """Convert entity annotations to IOB format"""
    conll_data = []
    
    for example in data:
        text = example['text']
        entities = example['entities']
        
        # Tokenize
        tokens = text.split()  # Simple tokenization
        
        # Create IOB labels
        labels = ['O'] * len(tokens)
        
        # Map entity spans to tokens
        for start, end, entity_type in entities:
            entity_text = text[start:end]
            entity_tokens = entity_text.split()
            
            # Find entity position in token list
            for i in range(len(tokens)):
                if tokens[i] == entity_tokens[0]:
                    labels[i] = f'B-{entity_type}'
                    for j in range(1, len(entity_tokens)):
                        if i+j < len(tokens):
                            labels[i+j] = f'I-{entity_type}'
        
        conll_data.append({
            'tokens': tokens,
            'labels': labels
        })
    
    return conll_data

# Convert data
conll_data = convert_to_conll(financial_data)

# Create Hugging Face dataset
from datasets import Dataset

dataset = Dataset.from_list(conll_data)

# Define label list
financial_labels = [
    'O',
    'B-COMPANY', 'I-COMPANY',
    'B-TICKER', 'I-TICKER',
    'B-MONEY', 'I-MONEY',
    'B-PERCENT', 'I-PERCENT',
    'B-QUARTER', 'I-QUARTER',
    'B-PERSON', 'I-PERSON'
]

# Train model (same process as above)
# ...
\`\`\`

## Production NER Pipeline

Building a production-ready NER system:

\`\`\`python
import torch
from typing import List, Dict, Tuple
from transformers import pipeline
import re

class ProductionNER:
    """Production-ready NER system with preprocessing and post-processing"""
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Entity confidence threshold
        self.confidence_threshold = 0.85
        
        # Entity type mapping
        self.entity_type_mapping = {
            'PER': 'PERSON',
            'LOC': 'LOCATION',
            'ORG': 'ORGANIZATION',
            'MISC': 'MISCELLANEOUS'
        }
    
    def preprocess(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Fix common issues
        text = text.strip()
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities with confidence filtering"""
        # Preprocess
        text = self.preprocess(text)
        
        # Extract entities
        entities = self.ner_pipeline(text)
        
        # Filter by confidence and standardize
        filtered_entities = []
        for ent in entities:
            if ent['score'] >= self.confidence_threshold:
                filtered_entities.append({
                    'text': ent['word'],
                    'type': self.entity_type_mapping.get(
                        ent['entity_group'],
                        ent['entity_group']
                    ),
                    'start': ent['start'],
                    'end': ent['end'],
                    'confidence': round(ent['score'], 3)
                })
        
        return filtered_entities
    
    def deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate/overlapping entities"""
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end']))
        
        # Remove overlaps (keep longer/higher confidence)
        deduplicated = [sorted_entities[0]]
        
        for ent in sorted_entities[1:]:
            last_ent = deduplicated[-1]
            
            # Check if overlapping
            if ent['start'] >= last_ent['end']:
                deduplicated.append(ent)
            elif ent['confidence'] > last_ent['confidence']:
                # Replace with higher confidence
                deduplicated[-1] = ent
        
        return deduplicated
    
    def group_by_type(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """Group entities by type"""
        grouped = {}
        for ent in entities:
            entity_type = ent['type']
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(ent['text'])
        
        return grouped
    
    def process_document(self, text: str) -> Dict:
        """Complete NER processing pipeline"""
        # Extract entities
        entities = self.extract_entities(text)
        
        # Deduplicate
        entities = self.deduplicate_entities(entities)
        
        # Group by type
        grouped = self.group_by_type(entities)
        
        return {
            'text': text,
            'entities': entities,
            'entity_counts': {k: len(v) for k, v in grouped.items()},
            'grouped_entities': grouped
        }

# Usage
ner = ProductionNER()

text = """
Apple Inc. announced its Q1 2024 earnings yesterday. CEO Tim Cook 
reported record revenue of $119.6 billion. The company's iPhone sales 
in China and Europe exceeded expectations. Apple's stock (AAPL) 
rose 5% to $180 on the NASDAQ exchange.
"""

result = ner.process_document(text)

print("Entities Found:")
for ent in result['entities']:
    print(f"  {ent['text']:20} | {ent['type']:15} | {ent['confidence']:.3f}")

print("\\nEntity Counts:")
for entity_type, count in result['entity_counts'].items():
    print(f"  {entity_type}: {count}")

print("\\nGrouped Entities:")
for entity_type, entities in result['grouped_entities'].items():
    print(f"  {entity_type}: {', '.join(entities)}")
\`\`\`

## Evaluation Metrics for NER

\`\`\`python
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score

# Example predictions and ground truth
y_true = [
    ['O', 'B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'O'],
    ['B-LOC', 'I-LOC', 'O', 'O', 'B-PER', 'O', 'O']
]

y_pred = [
    ['O', 'B-PER', 'I-PER', 'O', 'B-ORG', 'O', 'O'],  # Missed I-ORG
    ['B-LOC', 'O', 'O', 'O', 'B-PER', 'O', 'O']        # Missed I-LOC
]

# Entity-level metrics (strict)
print("Entity-level Metrics (seqeval):")
print(classification_report(y_true, y_pred))

# Detailed breakdown
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
print(f"F1: {f1_score(y_true, y_pred):.3f}")

# Token-level metrics
from sklearn.metrics import classification_report as sklearn_report

# Flatten to token level
y_true_flat = [label for seq in y_true for label in seq]
y_pred_flat = [label for seq in y_pred for label in seq]

print("\\nToken-level Metrics (sklearn):")
print(sklearn_report(y_true_flat, y_pred_flat))
\`\`\`

**Metrics Interpretation:**

\`\`\`
Entity-Level (seqeval):
- Strict: Entity is correct only if EXACT match (type and boundaries)
- Example: "New York" as B-LOC B-LOC is WRONG if predicted B-LOC O

Token-Level (sklearn):
- Lenient: Each token evaluated independently
- Example: "New York" as B-LOC B-LOC gets partial credit even if "York" is wrong

Always use entity-level metrics for NER evaluation!
\`\`\`

## Common Challenges in NER

### 1. Entity Boundary Detection

\`\`\`
Problem: "New York City" vs "New York" vs "City"
Solution: Use B- and I- tags carefully, consider context

Problem: "iPhone 15 Pro Max" - how many entities?
Solution: Define clear entity definitions in annotation guidelines
\`\`\`

### 2. Nested Entities

\`\`\`
Example: "Apple Inc.'s CEO Tim Cook"
- "Apple Inc.'s CEO Tim Cook" (PERSON with title)
- "Apple Inc." (ORGANIZATION)
- "Tim Cook" (PERSON)

Standard IOB cannot handle nested entities. Solutions:
- Separate models for each entity level
- Graph-based approaches
- Span-based models
\`\`\`

### 3. Entity Ambiguity

\`\`\`
"Apple reported strong earnings" - ORG (Apple Inc.)
"I ate an apple" - Not an entity (fruit)

"Paris announced new regulations" - GPE (city) or PER (Paris Hilton)?

Solution: Use context, training data diversity, entity linking
\`\`\`

### 4. Domain Adaptation

\`\`\`
General NER models fail on domain-specific entities:
- Medical: "Metformin 500mg" (DRUG + DOSAGE)
- Legal: "18 USC 242" (STATUTE)
- Financial: "AAPL 150C 2024-03-15" (OPTION)

Solution: Fine-tune on domain-specific data
\`\`\`

## Best Practices

1. **Data Quality**: High-quality annotations are critical
2. **Consistent Guidelines**: Clear entity definitions prevent ambiguity
3. **Context Window**: Ensure sufficient context for entity disambiguation
4. **Post-Processing**: Apply rules for common patterns (emails, dates, phone numbers)
5. **Confidence Thresholding**: Filter low-confidence predictions
6. **Entity Linking**: Link entities to knowledge bases for disambiguation
7. **Regular Updates**: Retrain on new entities and evolving language
8. **Monitoring**: Track entity distribution shifts in production

## Financial NLP Application

Using NER for financial document analysis:

\`\`\`python
# Extract financial entities from earnings calls
financial_text = """
Apple Inc. (NASDAQ: AAPL) reported Q1 FY2024 earnings with revenue 
of $119.6 billion, up 2% YoY. iPhone revenue was $69.7 billion. 
Services revenue hit $23.1 billion. CEO Tim Cook highlighted strong 
performance in India and Brazil. The company declared a dividend of 
$0.24 per share and authorized a $110 billion buyback program.
"""

# Custom financial NER
result = ner.process_document(financial_text)

# Post-process for financial entities
import re

def extract_financial_metrics(text):
    """Extract financial metrics using regex + NER"""
    metrics = {}
    
    # Extract revenue figures
    revenue_pattern = r'revenue[\\s\\w]*\\$([\\d.]+)\\s*(billion|million)'
    revenues = re.findall(revenue_pattern, text, re.IGNORECASE)
    metrics['revenues'] = [f"\${amount} {unit}" for amount, unit in revenues]
    
    # Extract percentages
    percent_pattern = r'([\\d.]+)%'
    percentages = re.findall(percent_pattern, text)
    metrics['percentages'] = [f"{p}%" for p in percentages]
    
    # Extract ticker
    ticker_pattern = r'\\(([A-Z]{1,5})\\)'
    tickers = re.findall(ticker_pattern, text)
    metrics['tickers'] = tickers
    
    return metrics

financial_metrics = extract_financial_metrics(financial_text)
print("\\nFinancial Metrics Extracted:")
for key, values in financial_metrics.items():
    print(f"  {key}: {values}")
\`\`\`

## Summary

Named Entity Recognition:
- Identifies and classifies entities in text
- Uses IOB tagging scheme for sequence labeling
- Transformer models (BERT) achieve state-of-the-art performance
- Requires careful attention to entity boundaries and types
- Critical for information extraction and document understanding
- Domain adaptation improves performance on specialized entities
- Entity-level evaluation (seqeval) is the standard

**Next Steps**: Apply NER in question answering and information retrieval systems.
`,
};
