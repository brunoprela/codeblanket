export const bertEncoderModels = {
  title: 'BERT and Encoder Models',
  id: 'bert-encoder-models',
  content: `
# BERT and Encoder Models

## Introduction

While GPT uses decoder-only architecture for generation, BERT (Bidirectional Encoder Representations from Transformers) uses encoder-only architecture for understanding. BERT can look at context from both left and right simultaneously, making it powerful for tasks requiring deep text understanding: classification, named entity recognition, question answering, and sentence similarity.

This section covers BERT's architecture, masked language modeling, fine-tuning approaches, BERT variants, and when to use encoders vs decoders.

### Why BERT Matters

**Bidirectional Context**: Sees entire sentence at once (vs left-to-right GPT)
**Transfer Learning**: Pre-train once, fine-tune for many tasks
**Sentence Embeddings**: Creates rich representations of text
**Understanding Tasks**: Excels at classification, NER, QA
**Efficiency**: Smaller models can match larger decoder-only models on understanding tasks

---

## BERT Architecture

### Encoder-Only Design

\`\`\`python
"""
BERT architecture: Stacked transformer encoders
"""

import torch
import torch.nn as nn

class BERTModel(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers
    
    Key differences from GPT:
    1. Encoder-only (no decoder)
    2. Bidirectional attention (sees full context)
    3. Different training objective (MLM + NSP)
    4. Designed for understanding, not generation
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()
        
        # Token, position, segment embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # For sentence pairs
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Pooler: Extract [CLS] representation
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1 = attend, 0 = ignore)
            token_type_ids: [batch, seq_len] (0 = sentence A, 1 = sentence B)
        
        Returns:
            sequence_output: [batch, seq_len, d_model]
            pooled_output: [batch, d_model]
        """
        batch_size, seq_len = input_ids.size()
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Segment IDs (default to 0 if not provided)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        position_emb = self.position_embedding(position_ids)
        segment_emb = self.segment_embedding(token_type_ids)
        
        # Combine embeddings
        embeddings = token_emb + position_emb + segment_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Attention mask: Expand to [batch, 1, 1, seq_len] for broadcasting
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0  # Mask out padding
        
        # Pass through encoder layers
        hidden = embeddings
        for layer in self.encoder_layers:
            hidden = layer(hidden, attention_mask)
        
        sequence_output = hidden  # All token representations
        
        # Pooled output: [CLS] token (first token)
        pooled_output = self.pooler(hidden[:, 0, :])
        
        return sequence_output, pooled_output

# BERT configurations
bert_configs = {
    "bert-base": {
        "params": "110M",
        "layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "d_ff": 3072
    },
    "bert-large": {
        "params": "340M",
        "layers": 24,
        "d_model": 1024,
        "n_heads": 16,
        "d_ff": 4096
    }
}

# Example usage
vocab_size = 30000
model = BERTModel(vocab_size, **bert_configs["bert-base"])

input_ids = torch.randint(0, vocab_size, (4, 128))  # Batch of 4, seq len 128
attention_mask = torch.ones(4, 128)  # All tokens are real (not padding)

sequence_output, pooled_output = model(input_ids, attention_mask)

print(f"Sequence output: {sequence_output.shape}")  # [4, 128, 768]
print(f"Pooled output: {pooled_output.shape}")      # [4, 768]
\`\`\`

---

## Masked Language Modeling (MLM)

### Pre-training Objective

\`\`\`python
"""
Masked Language Modeling: BERT's core training objective
"""

class MaskedLanguageModeling:
    """
    MLM: Randomly mask tokens and predict them
    
    Example:
    Input:  "The cat [MASK] on the [MASK]"
    Target: "The cat sat on the mat"
    Predict: [MASK] → "sat", [MASK] → "mat"
    """
    
    def __init__(self, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
    
    def create_masked_lm_predictions(self, tokens):
        """
        Mask tokens for MLM training
        
        Strategy:
        - 15% of tokens are chosen for masking
        - Of those 15%:
          - 80%: Replace with [MASK]
          - 10%: Replace with random token
          - 10%: Keep original token
        
        Why not 100% [MASK]?
        - Model would never see real tokens during pre-training
        - Fine-tuning tasks don't have [MASK] tokens
        - This mismatch hurts performance
        """
        
        masked_tokens = tokens.copy()
        labels = np.full_like(tokens, -100)  # -100 = ignore in loss
        
        # Choose 15% of positions to mask (excluding special tokens)
        mask_indices = []
        for i, token in enumerate(tokens):
            if token not in [self.tokenizer.cls_token_id, 
                           self.tokenizer.sep_token_id,
                           self.tokenizer.pad_token_id]:
                if np.random.random() < self.mask_prob:
                    mask_indices.append(i)
        
        for idx in mask_indices:
            # Set label (what to predict)
            labels[idx] = tokens[idx]
            
            # 80%: Replace with [MASK]
            if np.random.random() < 0.8:
                masked_tokens[idx] = self.mask_token_id
            
            # 10%: Replace with random token
            elif np.random.random() < 0.5:
                masked_tokens[idx] = np.random.randint(0, self.vocab_size)
            
            # 10%: Keep original (helps model learn representations)
            else:
                pass  # Keep original
        
        return masked_tokens, labels

# Example
sentence = "The cat sat on the mat"
tokens = tokenizer.encode(sentence)
print(f"Original: {tokens}")

masked_tokens, labels = mlm.create_masked_lm_predictions(tokens)
print(f"Masked: {masked_tokens}")
print(f"Labels: {labels}")

# Output might be:
# Original: [101, 1996, 4937, 2938, 2006, 1996, 13523, 102]
# Masked:   [101, 1996, 103,  2938, 103,  1996, 13523, 102]
#                         ^^^       ^^^
# Labels:   [-100, -100, 4937, -100, 2006, -100, -100, -100]

# Training MLM
class MLMTrainingHead(nn.Module):
    """
    Prediction head for masked tokens
    """
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden = self.dense(hidden_states)
        hidden = nn.functional.gelu(hidden)
        hidden = self.layer_norm(hidden)
        logits = self.decoder(hidden) + self.bias
        return logits

# Training loop
def train_mlm(model, dataloader, optimizer):
    """
    Train BERT with MLM objective
    """
    model.train()
    
    for batch in dataloader:
        # Get inputs
        input_ids = batch['input_ids']  # Already masked
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        
        # Forward pass
        sequence_output, _ = model(input_ids, attention_mask)
        prediction_scores = mlm_head(sequence_output)
        
        # Loss: Only on masked tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            prediction_scores.view(-1, model.vocab_size),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item()
\`\`\`

---

## Next Sentence Prediction (NSP)

### Understanding Sentence Relationships

\`\`\`python
"""
Next Sentence Prediction: Learn sentence relationships
"""

class NextSentencePrediction:
    """
    NSP: Given two sentences, predict if B follows A
    
    Example:
    Sentence A: "The cat sat on the mat"
    Sentence B: "It was very comfortable"
    Label: IsNext (1)
    
    Sentence A: "The cat sat on the mat"
    Sentence B: "The weather is nice today"
    Label: NotNext (0)
    """
    
    def create_nsp_training_examples(self, documents):
        """
        Create sentence pairs for NSP training
        """
        examples = []
        
        for doc in documents:
            sentences = split_into_sentences(doc)
            
            for i in range(len(sentences) - 1):
                sentence_a = sentences[i]
                
                # 50%: Use actual next sentence
                if np.random.random() < 0.5:
                    sentence_b = sentences[i + 1]
                    label = 1  # IsNext
                
                # 50%: Use random sentence from corpus
                else:
                    random_doc = random.choice(documents)
                    random_sentences = split_into_sentences(random_doc)
                    sentence_b = random.choice(random_sentences)
                    label = 0  # NotNext
                
                examples.append({
                    'sentence_a': sentence_a,
                    'sentence_b': sentence_b,
                    'label': label
                })
        
        return examples

# Input format for BERT
def prepare_nsp_input(sentence_a, sentence_b, tokenizer):
    """
    Format: [CLS] sentence_a [SEP] sentence_b [SEP]
    """
    tokens_a = tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.tokenize(sentence_b)
    
    # Build sequence
    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
    
    # Segment IDs (0 for sentence A, 1 for sentence B)
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    
    # Convert to IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return input_ids, segment_ids

# NSP prediction head
class NSPHead(nn.Module):
    """
    Binary classifier for next sentence prediction
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)  # Binary: IsNext or NotNext
    
    def forward(self, pooled_output):
        """
        Args:
            pooled_output: [batch, d_model] (from [CLS] token)
        
        Returns:
            logits: [batch, 2]
        """
        return self.classifier(pooled_output)

# Combined BERT pre-training
def pretrain_bert(model, dataloader, optimizer):
    """
    Train with both MLM and NSP
    """
    mlm_head = MLMTrainingHead(model.d_model, model.vocab_size)
    nsp_head = NSPHead(model.d_model)
    
    model.train()
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        mlm_labels = batch['mlm_labels']
        nsp_labels = batch['nsp_labels']
        
        # Forward pass
        sequence_output, pooled_output = model(
            input_ids, 
            attention_mask, 
            token_type_ids
        )
        
        # MLM loss
        mlm_scores = mlm_head(sequence_output)
        mlm_loss = cross_entropy(mlm_scores, mlm_labels)
        
        # NSP loss
        nsp_scores = nsp_head(pooled_output)
        nsp_loss = cross_entropy(nsp_scores, nsp_labels)
        
        # Combined loss
        loss = mlm_loss + nsp_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return mlm_loss.item(), nsp_loss.item()

# Note: Later research (RoBERTa) showed NSP is not very useful
# Modern models often skip NSP and only use MLM
\`\`\`

---

## Fine-Tuning BERT

### Task-Specific Fine-Tuning

\`\`\`python
"""
Fine-tune BERT for downstream tasks
"""

from transformers import BertForSequenceClassification, BertTokenizer, Trainer

# 1. Text Classification
class BERTClassifier:
    """
    Fine-tune BERT for classification
    """
    
    def __init__(self, num_labels):
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def prepare_data(self, texts, labels):
        """
        Tokenize texts for BERT
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }
    
    def train(self, train_texts, train_labels, val_texts, val_labels):
        """
        Fine-tune on classification task
        """
        # Prepare data
        train_data = self.prepare_data(train_texts, train_labels)
        val_data = self.prepare_data(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data
        )
        
        # Train
        trainer.train()
        
        return trainer
    
    def predict(self, texts):
        """
        Predict on new texts
        """
        self.model.eval()
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.numpy()

# Example: Sentiment analysis
classifier = BERTClassifier(num_labels=3)  # Positive, Negative, Neutral

train_texts = [
    "I love this product!",
    "Terrible experience, waste of money.",
    "It's okay, nothing special."
]
train_labels = [0, 1, 2]  # 0=Positive, 1=Negative, 2=Neutral

trainer = classifier.train(train_texts, train_labels, val_texts, val_labels)

# Predict
new_texts = ["This is amazing!", "Not great"]
predictions = classifier.predict(new_texts)
print(predictions)  # [0, 1] → Positive, Negative

# 2. Named Entity Recognition (NER)
from transformers import BertForTokenClassification

class BERTNER:
    """
    Fine-tune BERT for NER
    """
    
    def __init__(self, num_labels):
        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Label mapping
        self.id2label = {
            0: 'O',        # Outside
            1: 'B-PER',    # Person
            2: 'I-PER',
            3: 'B-ORG',    # Organization
            4: 'I-ORG',
            5: 'B-LOC',    # Location
            6: 'I-LOC'
        }
    
    def predict_entities(self, text):
        """
        Extract named entities
        """
        self.model.eval()
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Extract entities
        entities = []
        current_entity = []
        current_type = None
        
        for token, pred_id in zip(tokens, predictions[0][1:-1]):  # Skip [CLS] and [SEP]
            label = self.id2label[pred_id.item()]
            
            if label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append((current_type, ' '.join(current_entity)))
                current_entity = [token]
                current_type = label[2:]  # Remove 'B-'
            
            elif label.startswith('I-') and current_entity:
                # Continue entity
                current_entity.append(token)
            
            else:
                # Outside entity
                if current_entity:
                    entities.append((current_type, ' '.join(current_entity)))
                current_entity = []
                current_type = None
        
        return entities

# Example
ner = BERTNER(num_labels=7)
text = "John Smith works at Microsoft in Seattle."
entities = ner.predict_entities(text)
print(entities)
# [('PER', 'John Smith'), ('ORG', 'Microsoft'), ('LOC', 'Seattle')]

# 3. Question Answering
from transformers import BertForQuestionAnswering

class BERTQA:
    """
    Fine-tune BERT for extractive QA
    """
    
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad'
        )
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    
    def answer_question(self, question, context):
        """
        Extract answer span from context
        """
        # Encode question and context
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        # Predict start and end positions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # Get answer span
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Extract answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens)
        
        return answer

# Example
qa = BERTQA()
context = "BERT was published in 2018 by researchers at Google. It uses bidirectional training."
question = "When was BERT published?"
answer = qa.answer_question(question, context)
print(answer)  # "2018"
\`\`\`

---

## BERT Variants

### Improvements and Alternatives

\`\`\`python
"""
BERT variants and improvements
"""

# 1. RoBERTa (Robustly Optimized BERT)
class RoBERTa:
    """
    Key improvements:
    1. Remove NSP (not useful)
    2. Dynamic masking (different masks each epoch)
    3. More data (160GB vs 16GB)
    4. Larger batches
    5. Longer training
    
    Result: Better performance than BERT
    """
    
    def dynamic_masking(self, text):
        """
        Mask different tokens each epoch
        vs BERT's static masking
        """
        # Each epoch, create new masks
        # Exposes model to more variations
        pass

# 2. ALBERT (A Lite BERT)
class ALBERT:
    """
    Reduces parameters while maintaining performance
    
    Key techniques:
    1. Factorized embedding: Separate token embedding size from hidden size
    2. Cross-layer parameter sharing: Reuse weights across layers
    3. SOP instead of NSP: Sentence Order Prediction
    
    Result: 18x fewer parameters than BERT-large
    """
    
    def factorized_embedding(self):
        """
        V x H → V x E → E x H
        where V = vocab, H = hidden, E = embedding (E << H)
        
        Example:
        BERT: 30000 x 1024 = 30M parameters
        ALBERT: 30000 x 128 + 128 x 1024 = 4M parameters
        """
        vocab_embedding = nn.Embedding(30000, 128)
        projection = nn.Linear(128, 1024)
        
        # Embedding size 128 instead of 1024
        # Saves 26M parameters!

# 3. DistilBERT
class DistilBERT:
    """
    Knowledge distillation: Train small model to mimic large model
    
    Result: 40% smaller, 60% faster, retains 97% of BERT's performance
    """
    
    def distillation_loss(self, student_logits, teacher_logits, labels, temperature=2.0):
        """
        Combine two losses:
        1. Soft targets from teacher (knowledge transfer)
        2. Hard labels (ground truth)
        """
        # Soft loss: Match teacher's probability distribution
        soft_loss = nn.KLDivLoss()(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1)
        ) * (temperature ** 2)
        
        # Hard loss: Match true labels
        hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
        
        # Combined loss
        loss = 0.5 * soft_loss + 0.5 * hard_loss
        
        return loss

# 4. ELECTRA (Efficiently Learning an Encoder)
class ELECTRA:
    """
    More efficient pre-training
    
    Instead of MLM:
    1. Generator: Small model that corrupts tokens
    2. Discriminator: Predicts which tokens are corrupted
    
    Result: Better performance with less compute
    """
    
    def __init__(self):
        self.generator = BERTSmall()  # 1/4 size
        self.discriminator = BERT()
    
    def train_step(self, tokens):
        """
        1. Generator replaces masked tokens
        2. Discriminator detects replaced tokens
        """
        # Mask tokens
        masked_tokens, mask_indices = mask_tokens(tokens)
        
        # Generator predicts masked tokens
        generated_tokens = self.generator(masked_tokens)
        
        # Replace masked positions with generator predictions
        corrupted_tokens = tokens.copy()
        for idx in mask_indices:
            corrupted_tokens[idx] = generated_tokens[idx]
        
        # Discriminator: Binary classification for each token
        # 0 = original, 1 = replaced
        is_replaced = (corrupted_tokens != tokens).float()
        predictions = self.discriminator(corrupted_tokens)
        
        # Loss: Binary cross-entropy
        loss = binary_cross_entropy(predictions, is_replaced)
        
        return loss

# Model comparison
models = {
    "BERT-base": {
        "params": "110M",
        "training": "1M steps, 16GB data",
        "GLUE": "78.3%"
    },
    "RoBERTa-base": {
        "params": "125M",
        "training": "500K steps, 160GB data",
        "GLUE": "80.5%"
    },
    "ALBERT-xxlarge": {
        "params": "235M",  # Would be 1.5B if not for parameter sharing
        "training": "125K steps",
        "GLUE": "90.2%"
    },
    "DistilBERT": {
        "params": "66M",
        "training": "Distilled from BERT",
        "GLUE": "77.0%"
    },
    "ELECTRA-base": {
        "params": "110M",
        "training": "766K steps (more efficient)",
        "GLUE": "79.4%"
    }
}
\`\`\`

---

## When to Use BERT vs GPT

### Architecture Comparison

\`\`\`python
"""
Choosing between encoder (BERT) and decoder (GPT) models
"""

class ModelSelection:
    """
    Decision framework for architecture selection
    """
    
    def choose_model(self, task_type):
        """
        Task-based model selection
        """
        recommendations = {
            # Understanding tasks → BERT (encoder)
            "text_classification": "BERT",
            "sentiment_analysis": "BERT",
            "named_entity_recognition": "BERT",
            "question_answering": "BERT",
            "sentence_similarity": "BERT",
            "text_ranking": "BERT",
            
            # Generation tasks → GPT (decoder)
            "text_generation": "GPT",
            "code_generation": "GPT",
            "summarization": "GPT",  # Can also use encoder-decoder
            "translation": "GPT",    # Can also use encoder-decoder
            "dialogue": "GPT",
            "creative_writing": "GPT",
            
            # Either works (but GPT more popular now)
            "information_extraction": "BERT or GPT",
            "text_completion": "GPT",
        }
        
        return recommendations.get(task_type, "GPT (general purpose)")

# Example use cases

# Use BERT when:
# 1. Need deep understanding of text
# 2. Fixed-length output (classification, NER)
# 3. Smaller models acceptable
# 4. Bidirectional context critical

# Example: Spam detection
def spam_detection_bert():
    """
    BERT excels at classification
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    email = "URGENT!!! Click here for FREE MONEY!!!"
    prediction = model.predict(email)
    return prediction  # Spam

# Use GPT when:
# 1. Need to generate text
# 2. Variable-length output
# 3. Few-shot learning preferred
# 4. Multiple capabilities in one model

# Example: Email reply generation
def email_reply_gpt():
    """
    GPT excels at generation
    """
    email = "Can we reschedule tomorrow's meeting?"
    prompt = f"Email: {email}\\n\\nReply:"
    
    response = gpt_model.generate(prompt)
    return response
    # "Of course! What time works better for you?"

# Practical considerations
comparison = {
    "BERT": {
        "strengths": [
            "Best for understanding/classification",
            "Smaller models work well",
            "Efficient inference",
            "Strong sentence embeddings"
        ],
        "weaknesses": [
            "Can't generate text",
            "Requires fine-tuning for tasks",
            "Limited by max sequence length"
        ],
        "when_to_use": "Classification, NER, QA, ranking"
    },
    "GPT": {
        "strengths": [
            "Versatile (many tasks)",
            "Can generate text",
            "Few-shot learning",
            "No fine-tuning needed"
        ],
        "weaknesses": [
            "Larger models needed",
            "More expensive inference",
            "Unidirectional context"
        ],
        "when_to_use": "Generation, dialogue, diverse tasks"
    }
}

# Modern trend:
# - GPT-style models (decoder-only) dominating
# - But BERT still best for specific classification tasks
# - Encoder-decoder models (T5, BART) for seq2seq
\`\`\`

---

## Sentence Embeddings with BERT

### Creating Semantic Representations

\`\`\`python
"""
Using BERT for sentence embeddings
"""

from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

class BERTEmbeddings:
    """
    Generate sentence embeddings with BERT
    """
    
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.eval()
    
    def get_embedding(self, text):
        """
        Get embedding for a sentence
        
        Methods:
        1. [CLS] token (traditional)
        2. Mean pooling (better)
        3. Max pooling
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Method 1: [CLS] token
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Method 2: Mean pooling (recommended)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Mask padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        
        return mean_embedding.numpy()[0]
    
    def similarity(self, text1, text2):
        """
        Compute semantic similarity
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return sim

# Example: Semantic search
bert_emb = BERTEmbeddings()

documents = [
    "The cat sat on the mat",
    "A feline rested on a rug",
    "The weather is nice today",
    "Dogs are loyal animals"
]

query = "A cat on a mat"

# Get embeddings
query_emb = bert_emb.get_embedding(query)
doc_embs = [bert_emb.get_embedding(doc) for doc in documents]

# Compute similarities
similarities = cosine_similarity([query_emb], doc_embs)[0]

# Rank documents
ranked = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

for doc, sim in ranked:
    print(f"{sim:.3f}: {doc}")

# Output:
# 0.92: The cat sat on the mat (exact match)
# 0.87: A feline rested on a rug (semantic match)
# 0.45: Dogs are loyal animals (somewhat related)
# 0.32: The weather is nice today (unrelated)

# Sentence-BERT (SBERT): Better embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Much better embeddings for similarity tasks
embeddings = model.encode(documents)
query_emb = model.encode(query)

similarities = cosine_similarity([query_emb], embeddings)[0]
# Even better semantic matching!
\`\`\`

---

## Conclusion

BERT and encoder models:

1. **Bidirectional Context**: See full sentence (vs left-to-right GPT)
2. **Masked Language Modeling**: Learn by predicting masked tokens
3. **Understanding Tasks**: Excel at classification, NER, QA
4. **Sentence Embeddings**: Create rich semantic representations
5. **Fine-Tuning**: Quick adaptation to specific tasks

**Key Variants**:
- **RoBERTa**: Better training, no NSP
- **ALBERT**: Parameter-efficient with sharing
- **DistilBERT**: Fast, small, distilled
- **ELECTRA**: Efficient discriminative training

**Practical Takeaways**:
- Use BERT for classification and understanding
- Use GPT for generation and diverse tasks
- Sentence-BERT for embeddings and similarity
- RoBERTa or ELECTRA for best performance
- DistilBERT for speed/size constraints

While GPT-style models dominate headlines, BERT remains highly effective for classification and understanding tasks, especially when labeled data is available for fine-tuning.
`,
};
