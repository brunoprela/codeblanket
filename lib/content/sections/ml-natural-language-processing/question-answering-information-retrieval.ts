/**
 * Section: Question Answering & Information Retrieval
 * Module: Natural Language Processing
 *
 * Covers extractive QA, semantic search, dense retrieval, and information retrieval systems
 */

export const questionAnsweringSection = {
  id: 'question-answering-information-retrieval',
  title: 'Question Answering & Information Retrieval',
  content: `
# Question Answering & Information Retrieval

## Introduction

Question Answering (QA) systems find answers to natural language questions, while Information Retrieval (IR) finds relevant documents. Together, they enable search engines, chatbots, and knowledge retrieval systems.

**Key Applications:**
- **Search engines**: Google, Bing semantic search
- **Chatbots**: Customer service, virtual assistants
- **Document QA**: Finding answers in long documents
- **Knowledge bases**: Wikipedia search, enterprise knowledge

## Extractive Question Answering

Extractive QA finds answer spans within given context (reading comprehension).

### Using Pre-trained QA Models

\`\`\`python
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Using pipeline (simplest)
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
Constructed from 1887 to 1889, it was initially criticized by some of France\'s leading artists 
and intellectuals for its design, but has become a global cultural icon of France and one of 
the most recognizable structures in the world.
"""

question = "When was the Eiffel Tower built?"

result = qa_pipeline (question=question, context=context)
print(f"Answer: {result['answer']}")
print(f"Score: {result['score']:.4f}")
print(f"Start: {result['start']}, End: {result['end']}")

# Output:
# Answer: from 1887 to 1889
# Score: 0.9856
# Start: 152, End: 171
\`\`\`

### Manual QA with Model

\`\`\`python
import torch

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def answer_question (question, context):
    """Extract answer span from context"""
    # Encode question and context
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    
    # Get start and end logits
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # Get most likely start and end positions
    start_index = torch.argmax (start_logits)
    end_index = torch.argmax (end_logits)
    
    # Decode tokens to get answer
    tokens = tokenizer.convert_ids_to_tokens (inputs['input_ids'][0])
    answer_tokens = tokens[start_index:end_index+1]
    answer = tokenizer.convert_tokens_to_string (answer_tokens)
    
    # Calculate confidence score
    start_score = torch.max (torch.softmax (start_logits, dim=1)).item()
    end_score = torch.max (torch.softmax (end_logits, dim=1)).item()
    confidence = (start_score + end_score) / 2
    
    return {
        'answer': answer,
        'confidence': confidence,
        'start_token': start_index.item(),
        'end_token': end_index.item()
    }

# Example
question = "Who designed the Eiffel Tower?"
result = answer_question (question, context)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.4f}")
\`\`\`

### Fine-tuning on SQuAD

\`\`\`python
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)

# Load SQuAD dataset
datasets = load_dataset('squad')

# Tokenize and prepare features
def prepare_train_features (examples):
    # Tokenize questions and contexts
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )
    
    # Map answer positions to token positions
    offset_mapping = tokenized_examples.pop('offset_mapping')
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []
    
    for i, offsets in enumerate (offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index (tokenizer.cls_token_id)
        
        sequence_ids = tokenized_examples.sequence_ids (i)
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        
        # If no answer, set to CLS index
        if len (answers['answer_start']) == 0:
            tokenized_examples['start_positions'].append (cls_index)
            tokenized_examples['end_positions'].append (cls_index)
        else:
            start_char = answers['answer_start'][0]
            end_char = start_char + len (answers['text'][0])
            
            # Find token start position
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            # Find token end position
            token_end_index = len (input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # Check if answer is within context
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples['start_positions'].append (cls_index)
                tokenized_examples['end_positions'].append (cls_index)
            else:
                # Find start and end token positions
                while token_start_index < len (offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples['start_positions'].append (token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples['end_positions'].append (token_end_index + 1)
    
    return tokenized_examples

tokenized_datasets = datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=datasets['train'].column_names
)

# Training
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./qa_model',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=default_data_collator,
)

trainer.train()
\`\`\`

## Information Retrieval

### Traditional IR: BM25

\`\`\`python
from rank_bm25 import BM25Okapi
import numpy as np

# Document corpus
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing enables computers to understand text",
    "Deep learning uses neural networks with multiple layers",
    "Python is a popular programming language for data science"
]

# Tokenize documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Create BM25 index
bm25 = BM25Okapi (tokenized_docs)

# Query
query = "machine learning artificial intelligence"
tokenized_query = query.lower().split()

# Get scores
scores = bm25.get_scores (tokenized_query)
print("BM25 Scores:", scores)

# Get top k documents
top_k = 3
top_docs = np.argsort (scores)[::-1][:top_k]

print(f"\\nQuery: {query}")
print("Top results:")
for i, doc_idx in enumerate (top_docs):
    print(f"{i+1}. {documents[doc_idx]} (score: {scores[doc_idx]:.4f})")
\`\`\`

### Dense Retrieval with Sentence Embeddings

\`\`\`python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing enables computers to understand text",
    "Deep learning uses neural networks with multiple layers",
    "Python is a popular programming language for data science"
]

doc_embeddings = model.encode (documents)
print(f"Document embeddings shape: {doc_embeddings.shape}")  # (5, 384)

# Encode query
query = "What is machine learning?"
query_embedding = model.encode([query])

# Compute similarity
similarities = cosine_similarity (query_embedding, doc_embeddings)[0]
print(f"\\nQuery: {query}")
print("Similarities:", similarities)

# Get top k
top_k = 3
top_indices = np.argsort (similarities)[::-1][:top_k]

print("\\nTop results:")
for i, idx in enumerate (top_indices):
    print(f"{i+1}. {documents[idx]}")
    print(f"   Similarity: {similarities[idx]:.4f}")
\`\`\`

### Hybrid Search (BM25 + Dense)

\`\`\`python
def hybrid_search (query, documents, alpha=0.5):
    """
    Combine BM25 (sparse) and dense retrieval
    alpha: weight for dense scores (1-alpha for BM25)
    """
    # BM25 scores
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi (tokenized_docs)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores (tokenized_query)
    
    # Dense scores
    doc_embeddings = model.encode (documents)
    query_embedding = model.encode([query])
    dense_scores = cosine_similarity (query_embedding, doc_embeddings)[0]
    
    # Normalize scores to [0, 1]
    bm25_normalized = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
    dense_normalized = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-10)
    
    # Combine
    hybrid_scores = alpha * dense_normalized + (1 - alpha) * bm25_normalized
    
    return hybrid_scores

# Example
query = "machine learning AI"
hybrid_scores = hybrid_search (query, documents, alpha=0.5)

print(f"Query: {query}")
print("\\nHybrid scores:")
for i, score in enumerate (hybrid_scores):
    print(f"{documents[i]}: {score:.4f}")
\`\`\`

## Dense Passage Retrieval (DPR)

\`\`\`python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import torch

# Load DPR models
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Encode passages
passages = [
    "Paris is the capital of France and its most populous city",
    "London is the capital of England and the United Kingdom",
    "Berlin is the capital and largest city of Germany",
    "Madrid is the capital of Spain"
]

context_inputs = context_tokenizer (passages, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    context_embeddings = context_encoder(**context_inputs).pooler_output

# Encode question
question = "What is the capital of France?"
question_inputs = question_tokenizer (question, return_tensors='pt')
with torch.no_grad():
    question_embedding = question_encoder(**question_inputs).pooler_output

# Compute similarities
similarities = torch.matmul (question_embedding, context_embeddings.T)[0]
print(f"\\nQuestion: {question}")
print("Similarities:")
for i, sim in enumerate (similarities):
    print(f"{passages[i]}: {sim.item():.4f}")

# Get best match
best_idx = torch.argmax (similarities).item()
print(f"\\nBest match: {passages[best_idx]}")
\`\`\`

## Building a Complete QA System

\`\`\`python
class QASystem:
    """Complete Question Answering system with retrieval"""
    
    def __init__(self, retriever_model='all-MiniLM-L6-v2', qa_model='distilbert-base-cased-distilled-squad'):
        # Retriever
        self.retriever = SentenceTransformer (retriever_model)
        
        # QA model
        self.qa_pipeline = pipeline('question-answering', model=qa_model)
        
        # Document store
        self.documents = []
        self.document_embeddings = None
    
    def add_documents (self, documents):
        """Index documents"""
        self.documents = documents
        self.document_embeddings = self.retriever.encode (documents)
        print(f"Indexed {len (documents)} documents")
    
    def retrieve (self, question, top_k=3):
        """Retrieve relevant documents"""
        question_embedding = self.retriever.encode([question])
        similarities = cosine_similarity (question_embedding, self.document_embeddings)[0]
        
        top_indices = np.argsort (similarities)[::-1][:top_k]
        return [(self.documents[i], similarities[i]) for i in top_indices]
    
    def answer (self, question, top_k=3):
        """Retrieve documents and extract answer"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve (question, top_k)
        
        print(f"\\nQuestion: {question}")
        print(f"\\nRetrieved {len (retrieved_docs)} documents:")
        for i, (doc, score) in enumerate (retrieved_docs):
            print(f"{i+1}. (score: {score:.4f}) {doc[:100]}...")
        
        # Try to answer from each document
        answers = []
        for doc, retrieval_score in retrieved_docs:
            try:
                result = self.qa_pipeline (question=question, context=doc)
                answers.append({
                    'answer': result['answer'],
                    'qa_score': result['score'],
                    'retrieval_score': retrieval_score,
                    'combined_score': result['score'] * retrieval_score,
                    'context': doc
                })
            except:
                continue
        
        if not answers:
            return None
        
        # Return best answer by combined score
        best_answer = max (answers, key=lambda x: x['combined_score'])
        return best_answer

# Example usage
qa_system = QASystem()

# Add documents
documents = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.",
    "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor. It was a gift from France to the United States.",
    "The Great Wall of China is a series of fortifications built across northern China. It was built to protect Chinese states from invasions.",
    "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, India.",
]

qa_system.add_documents (documents)

# Ask questions
question = "Who was the Eiffel Tower named after?"
result = qa_system.answer (question)

if result:
    print(f"\\nAnswer: {result['answer']}")
    print(f"QA Score: {result['qa_score']:.4f}")
    print(f"Retrieval Score: {result['retrieval_score']:.4f}")
    print(f"Combined Score: {result['combined_score']:.4f}")
\`\`\`

## Semantic Search with FAISS

\`\`\`python
import faiss
import numpy as np

# Create embeddings
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "NLP enables computers to understand text",
    "Python is popular for data science",
    "Transformers revolutionized NLP"
] * 1000  # Scale up

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode (documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add (embeddings.astype('float32'))

print(f"Indexed {index.ntotal} documents")

# Search
query = "What is machine learning?"
query_embedding = model.encode([query]).astype('float32')

k = 5  # Top 5 results
distances, indices = index.search (query_embedding, k)

print(f"\\nQuery: {query}")
print("Results:")
for i, (dist, idx) in enumerate (zip (distances[0], indices[0])):
    print(f"{i+1}. {documents[idx]} (distance: {dist:.4f})")
\`\`\`

## Evaluation Metrics

\`\`\`python
# For retrieval: Precision@K, Recall@K, MAP, MRR

def precision_at_k (relevant, retrieved, k):
    """Precision at K"""
    retrieved_k = retrieved[:k]
    relevant_retrieved = len (set (relevant) & set (retrieved_k))
    return relevant_retrieved / k

def recall_at_k (relevant, retrieved, k):
    """Recall at K"""
    retrieved_k = retrieved[:k]
    relevant_retrieved = len (set (relevant) & set (retrieved_k))
    return relevant_retrieved / len (relevant)

def average_precision (relevant, retrieved):
    """Average Precision"""
    score = 0.0
    num_relevant = 0
    
    for i, doc in enumerate (retrieved):
        if doc in relevant:
            num_relevant += 1
            score += num_relevant / (i + 1)
    
    return score / len (relevant) if relevant else 0.0

def mean_average_precision (relevance_lists, retrieval_lists):
    """MAP across multiple queries"""
    return np.mean([average_precision (rel, ret) 
                   for rel, ret in zip (relevance_lists, retrieval_lists)])

# For QA: Exact Match (EM) and F1

def normalize_answer (s):
    """Normalize answer for comparison"""
    import re
    import string
    
    def remove_articles (text):
        return re.sub (r'\\b (a|an|the)\\b', ' ', text)
    
    def white_space_fix (text):
        return ' '.join (text.split())
    
    def remove_punc (text):
        exclude = set (string.punctuation)
        return '.join (ch for ch in text if ch not in exclude)
    
    def lower (text):
        return text.lower()
    
    return white_space_fix (remove_articles (remove_punc (lower (s))))

def exact_match_score (prediction, ground_truth):
    """Exact match"""
    return normalize_answer (prediction) == normalize_answer (ground_truth)

def f1_score_qa (prediction, ground_truth):
    """Token-level F1"""
    pred_tokens = normalize_answer (prediction).split()
    gt_tokens = normalize_answer (ground_truth).split()
    
    common = set (pred_tokens) & set (gt_tokens)
    
    if len (common) == 0:
        return 0.0
    
    precision = len (common) / len (pred_tokens)
    recall = len (common) / len (gt_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

# Example
prediction = "Gustave Eiffel"
ground_truth = "the engineer Gustave Eiffel"

em = exact_match_score (prediction, ground_truth)
f1 = f1_score_qa (prediction, ground_truth)

print(f"Exact Match: {em}")
print(f"F1 Score: {f1:.4f}")
\`\`\`

## Summary

Question Answering and Information Retrieval:
- **Extractive QA**: Find answer spans in context
- **Dense retrieval**: Semantic search with embeddings
- **Hybrid search**: Combine sparse (BM25) and dense methods
- **End-to-end systems**: Retrieval + extraction
- **Evaluation**: EM, F1 for QA; Precision, Recall, MAP for IR

Modern systems combine powerful retrievers with reader models for accurate answers.
`,
};
