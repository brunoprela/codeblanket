/**
 * Section: Named Entity Recognition
 * Module: Natural Language Processing
 *
 * Covers NER, token classification, and entity extraction
 */

export const namedEntityRecognitionSection = {
  id: 'named-entity-recognition',
  title: 'Named Entity Recognition',
  content: `
# Named Entity Recognition (NER)

## Introduction

NER identifies and classifies named entities (people, organizations, locations, dates, etc.) in text. It's a sequence labeling task where each token gets a label.

## NER with Transformers

\`\`\`python
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

# Using pre-trained NER model
ner = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')
text = "Apple Inc. is located in Cupertino, California"
entities = ner(text)

for entity in entities:
    print(f"{entity['word']}: {entity['entity']} (score: {entity['score']:.3f})")

# Output:
# Apple: B-ORG (score: 0.998)
# Inc: I-ORG (score: 0.995)
# Cupertino: B-LOC (score: 0.999)
# California: B-LOC (score: 0.998)
\`\`\`

## IOB Tagging Scheme

\`\`\`
B- (Begin): First token of entity
I- (Inside): Continuation of entity
O (Outside): Not an entity

Example: "Apple Inc. announced new iPhone"
Apple    B-ORG
Inc.     I-ORG
announced O
new      O
iPhone   B-PRODUCT
\`\`\`

## Fine-tuning for Custom NER

\`\`\`python
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load CoNLL-2003 dataset
dataset = load_dataset('conll2003')

# Labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
label_list = dataset['train'].features['ner_tags'].feature.names
num_labels = len(label_list)

# Load model
model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore padding
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignore subword tokens
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./ner_results', num_train_epochs=3),
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()
\`\`\`

## Evaluation

\`\`\`python
from seqeval.metrics import classification_report, f1_score

# Predict
predictions = trainer.predict(tokenized_datasets['test'])
preds = np.argmax(predictions.predictions, axis=2)

# Convert to labels
true_labels = [[label_list[l] for l in label if l != -100] for label in predictions.label_ids]
pred_labels = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
               for prediction, label in zip(preds, predictions.label_ids)]

# seqeval provides entity-level metrics
print(classification_report(true_labels, pred_labels))
\`\`\`

## Summary

NER identifies and classifies named entities using token-level classification with transformers.
`,
};
