/**
 * Section: Text Representation
 * Module: Natural Language Processing
 *
 * Covers Bag of Words, TF-IDF, N-grams, and count-based text representation methods
 */

export const textRepresentationSection = {
  id: 'text-representation',
  title: 'Text Representation',
  content: `
# Text Representation

## Introduction

Machine learning models work with numbers, not text. Text representation (also called text vectorization or text encoding) is the process of converting text into numerical format that algorithms can process. The quality of your text representation directly impacts model performance—a good representation captures semantic meaning while remaining computationally efficient.

**Why Text Representation Matters:**
- **Models need numbers**: Neural networks and ML algorithms require numerical input
- **Semantic capture**: Good representations preserve meaning and relationships
- **Dimensionality**: Balance between information retention and computational cost
- **Task-dependent**: Different tasks benefit from different representations

This section covers **count-based methods** (Bag of Words, TF-IDF, N-grams). Later sections will cover dense representations (word embeddings).

## Bag of Words (BoW)

The Bag of Words model represents text as an unordered collection of words, disregarding grammar and word order but keeping multiplicity (word frequency).

### Conceptual Understanding

\`\`\`python
# Imagine three documents
documents = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is amazing"
]

# Step 1: Build vocabulary (unique words)
vocabulary = {"I", "love", "machine", "learning", "deep", "is", "amazing"}
# Vocabulary size: 7 words

# Step 2: Represent each document as word counts
# Document 1: "I love machine learning"
# [I:1, love:1, machine:1, learning:1, deep:0, is:0, amazing:0]
# As vector: [1, 1, 1, 1, 0, 0, 0]

# Document 2: "I love deep learning"  
# [I:1, love:1, machine:0, learning:1, deep:1, is:0, amazing:0]
# As vector: [1, 1, 0, 1, 1, 0, 0]

# Document 3: "Machine learning is amazing"
# [I:0, love:0, machine:1, learning:1, deep:0, is:1, amazing:1]
# As vector: [0, 0, 1, 1, 0, 1, 1]
\`\`\`

### Implementation with scikit-learn

\`\`\`python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

documents = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is amazing",
    "Deep learning and machine learning are related"
]

# Create CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform
bow_matrix = vectorizer.fit_transform (documents)

# View vocabulary
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary ({len (vocab)} words): {vocab}")
# ['amazing', 'and', 'are', 'deep', 'is', 'learning', 'love', 'machine', 'related']

# View as DataFrame for clarity
bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vocab
)
print("\\nBag of Words Matrix:")
print(bow_df)
\`\`\`

**Output:**
\`\`\`
Vocabulary (9 words): ['amazing' 'and' 'are' 'deep' 'is' 'learning' 'love' 'machine' 'related']

Bag of Words Matrix:
   amazing  and  are  deep  is  learning  love  machine  related
0        0    0    0     0   0         1     1        1        0
1        0    0    0     1   0         1     1        0        0
2        1    0    0     0   1         1     0        1        0
3        0    1    1     1   0         2     0        1        1
\`\`\`

### BoW Parameters and Options

\`\`\`python
# Binary mode (presence/absence instead of count)
vectorizer_binary = CountVectorizer (binary=True)
bow_binary = vectorizer_binary.fit_transform (documents)
print("Binary BoW:")
print(pd.DataFrame (bow_binary.toarray(), columns=vectorizer_binary.get_feature_names_out()))

# Lowercasing (default=True)
vectorizer_case = CountVectorizer (lowercase=False)
# This would treat "Machine" and "machine" as different words

# Custom tokenization
import re
def custom_tokenizer (text):
    return re.findall (r'\\w+', text.lower())

vectorizer_custom = CountVectorizer (tokenizer=custom_tokenizer)

# Min/max document frequency (filter rare/common words)
vectorizer_filtered = CountVectorizer(
    min_df=2,  # Ignore words appearing in < 2 documents
    max_df=0.8  # Ignore words appearing in > 80% of documents
)

# Max features (keep only top N most frequent)
vectorizer_limited = CountVectorizer (max_features=1000)

# Stop words
vectorizer_no_stop = CountVectorizer (stop_words='english')

# Example with filtering
docs_large = documents * 10  # Simulate larger corpus
vectorizer_filtered.fit (docs_large)
print(f"\\nFiltered vocabulary size: {len (vectorizer_filtered.vocabulary_)}")
\`\`\`

### BoW from Scratch

\`\`\`python
from collections import Counter
import numpy as np

class SimpleBagOfWords:
    """Bag of Words implementation from scratch"""
    
    def __init__(self):
        self.vocabulary = {}
        self.inverse_vocabulary = {}
    
    def fit (self, documents):
        """Build vocabulary from documents"""
        # Collect all unique words
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update (words)
        
        # Create vocabulary mapping
        self.vocabulary = {word: idx for idx, word in enumerate (sorted (all_words))}
        self.inverse_vocabulary = {idx: word for word, idx in self.vocabulary.items()}
        
        return self
    
    def transform (self, documents):
        """Transform documents to BoW vectors"""
        vectors = []
        
        for doc in documents:
            # Initialize zero vector
            vector = np.zeros (len (self.vocabulary))
            
            # Count words
            words = doc.lower().split()
            word_counts = Counter (words)
            
            # Fill vector
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    vector[idx] = count
            
            vectors.append (vector)
        
        return np.array (vectors)
    
    def fit_transform (self, documents):
        """Fit and transform in one step"""
        self.fit (documents)
        return self.transform (documents)

# Test custom implementation
simple_bow = SimpleBagOfWords()
vectors = simple_bow.fit_transform (documents)

print("Custom BoW:")
print("Vocabulary:", simple_bow.vocabulary)
print("Vectors shape:", vectors.shape)
print("\\nFirst document vector:")
print(vectors[0])
\`\`\`

## TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF improves on BoW by considering both term frequency (how often a word appears in a document) and inverse document frequency (how rare/common a word is across all documents). It down-weights common words and up-weights rare, informative words.

### Mathematical Foundation

**Term Frequency (TF):**
- Measures how frequently a term appears in a document
- \`TF(t, d) = count of term t in document d / total terms in document d\`

**Inverse Document Frequency (IDF):**
- Measures how rare/common a term is across all documents  
- \`IDF(t) = log (total documents / documents containing term t)\`

**TF-IDF:**
- \`TF-IDF(t, d) = TF(t, d) × IDF(t)\`

**Intuition:**
- Common words (appears in many docs): Low IDF → Low TF-IDF
- Rare informative words (appears in few docs): High IDF → High TF-IDF
- Frequent in specific doc: High TF → High TF-IDF

### Implementation with scikit-learn

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are enemies",
    "The mat was comfortable"
]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform
tfidf_matrix = tfidf_vectorizer.fit_transform (documents)

# View as DataFrame
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("TF-IDF Matrix:")
print(tfidf_df.round(3))

# Get IDF values for each word
idf_values = pd.DataFrame({
    'word': tfidf_vectorizer.get_feature_names_out(),
    'idf': tfidf_vectorizer.idf_
}).sort_values('idf', ascending=False)

print("\\nIDF values (higher = more rare/informative):")
print(idf_values)
\`\`\`

**Output Analysis:**
\`\`\`
Words like "the" appear in many documents → Low IDF → Low TF-IDF
Words like "comfortable" appear in one document → High IDF → High TF-IDF (when present)
\`\`\`

### TF-IDF Parameters

\`\`\`python
# Different normalization
tfidf_l2 = TfidfVectorizer (norm='l2')  # L2 normalization (default)
tfidf_l1 = TfidfVectorizer (norm='l1')  # L1 normalization
tfidf_none = TfidfVectorizer (norm=None)  # No normalization

# Use IDF
tfidf_no_idf = TfidfVectorizer (use_idf=False)  # Just term frequency

# Smooth IDF (add 1 to document frequencies)
tfidf_smooth = TfidfVectorizer (smooth_idf=True)  # Default

# Sublinear TF (use log of term frequency)
tfidf_sublinear = TfidfVectorizer (sublinear_tf=True)  # TF = 1 + log(TF)

# Complete example
tfidf_advanced = TfidfVectorizer(
    lowercase=True,
    max_features=5000,
    min_df=2,
    max_df=0.8,
    stop_words='english',
    ngram_range=(1, 2),  # Unigrams and bigrams
    sublinear_tf=True,
    norm='l2'
)
\`\`\`

### TF-IDF from Scratch

\`\`\`python
import numpy as np
from collections import Counter
import math

class SimpleTfidf:
    """TF-IDF implementation from scratch"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.document_count = 0
    
    def fit (self, documents):
        """Calculate IDF values"""
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = set (doc.lower().split())  # Unique words per doc
            all_words.update (words)
        
        self.vocabulary = {word: idx for idx, word in enumerate (sorted (all_words))}
        self.document_count = len (documents)
        
        # Calculate document frequency for each word
        doc_frequency = Counter()
        for doc in documents:
            unique_words = set (doc.lower().split())
            for word in unique_words:
                doc_frequency[word] += 1
        
        # Calculate IDF
        for word in self.vocabulary:
            df = doc_frequency[word]
            # IDF = log (total docs / docs containing word)
            self.idf[word] = math.log (self.document_count / df)
        
        return self
    
    def transform (self, documents):
        """Transform documents to TF-IDF vectors"""
        vectors = []
        
        for doc in documents:
            # Initialize zero vector
            vector = np.zeros (len (self.vocabulary))
            
            # Calculate term frequency
            words = doc.lower().split()
            total_words = len (words)
            word_counts = Counter (words)
            
            # Calculate TF-IDF
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf = count / total_words  # Normalized term frequency
                    idf = self.idf[word]
                    vector[idx] = tf * idf
            
            # L2 normalization
            norm = np.linalg.norm (vector)
            if norm > 0:
                vector = vector / norm
            
            vectors.append (vector)
        
        return np.array (vectors)
    
    def fit_transform (self, documents):
        """Fit and transform in one step"""
        self.fit (documents)
        return self.transform (documents)

# Test custom TF-IDF
simple_tfidf = SimpleTfidf()
tfidf_vectors = simple_tfidf.fit_transform (documents)

print("Custom TF-IDF:")
print("Shape:", tfidf_vectors.shape)
print("\\nIDF values:")
for word, idf in sorted (simple_tfidf.idf.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {word}: {idf:.3f}")
\`\`\`

### BoW vs TF-IDF Comparison

\`\`\`python
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
docs = [
    "machine learning is a subset of artificial intelligence",
    "deep learning is a subset of machine learning",
    "the cat sat on the mat",
]

# BoW representation
bow_vec = CountVectorizer()
bow_matrix = bow_vec.fit_transform (docs)

# TF-IDF representation
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform (docs)

# Compare similarities
print("BoW Similarity Matrix:")
bow_sim = cosine_similarity (bow_matrix)
print(bow_sim.round(3))

print("\\nTF-IDF Similarity Matrix:")
tfidf_sim = cosine_similarity (tfidf_matrix)
print(tfidf_sim.round(3))

# Analysis
print("\\nAnalysis:")
print(f"Doc 0 vs Doc 1 (both about ML) - BoW: {bow_sim[0,1]:.3f}, TF-IDF: {tfidf_sim[0,1]:.3f}")
print(f"Doc 0 vs Doc 2 (different topics) - BoW: {bow_sim[0,2]:.3f}, TF-IDF: {tfidf_sim[0,2]:.3f}")
print("\\nTF-IDF better captures semantic similarity by down-weighting common words!")
\`\`\`

## N-grams

N-grams are contiguous sequences of n words. They capture local word order and phrases, which BoW ignores.

**Types:**
- **Unigrams (1-gram)**: Individual words ["machine", "learning"]
- **Bigrams (2-gram)**: Pairs of consecutive words ["machine learning", "learning is"]
- **Trigrams (3-gram)**: Triples ["machine learning is"]
- **N-grams**: Sequences of n words

### Why N-grams Matter

\`\`\`python
# Unigrams lose meaning
text1 = "not good"
text2 = "good"

# With unigrams only:
# text1 = ["not", "good"]
# text2 = ["good"]
# Both contain "good" - very similar!

# With bigrams:
# text1 = ["not good"]  # Negative phrase
# text2 = ["good"]      # Positive word
# Now captured as different!
\`\`\`

### Implementing N-grams

\`\`\`python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "machine learning is awesome",
    "deep learning is amazing",
    "machine learning and deep learning"
]

# Unigrams only (default)
unigram_vec = CountVectorizer (ngram_range=(1, 1))
unigram_matrix = unigram_vec.fit_transform (documents)
print(f"Unigrams ({unigram_matrix.shape[1]} features):")
print(unigram_vec.get_feature_names_out())

# Bigrams only
bigram_vec = CountVectorizer (ngram_range=(2, 2))
bigram_matrix = bigram_vec.fit_transform (documents)
print(f"\\nBigrams ({bigram_matrix.shape[1]} features):")
print(bigram_vec.get_feature_names_out())

# Unigrams + Bigrams
combined_vec = CountVectorizer (ngram_range=(1, 2))
combined_matrix = combined_vec.fit_transform (documents)
print(f"\\nUnigrams + Bigrams ({combined_matrix.shape[1]} features):")
print(combined_vec.get_feature_names_out())

# Unigrams + Bigrams + Trigrams
trigram_vec = CountVectorizer (ngram_range=(1, 3))
trigram_matrix = trigram_vec.fit_transform (documents)
print(f"\\nUp to Trigrams ({trigram_matrix.shape[1]} features):")

# Character n-grams (useful for typos, morphology)
char_ngram_vec = CountVectorizer(
    analyzer='char',  # Character-level
    ngram_range=(2, 4)  # 2-4 character sequences
)
char_matrix = char_ngram_vec.fit_transform (documents)
print(f"\\nCharacter n-grams ({char_matrix.shape[1]} features):")
print(char_ngram_vec.get_feature_names_out()[:20], "...")
\`\`\`

### TF-IDF with N-grams

\`\`\`python
# TF-IDF with bigrams
tfidf_bigram = TfidfVectorizer (ngram_range=(1, 2), max_features=50)

docs_sentiment = [
    "This movie is not good at all",
    "This movie is very good",
    "I did not like this movie",
    "I really liked this movie"
]

tfidf_bigram_matrix = tfidf_bigram.fit_transform (docs_sentiment)

# View important features
feature_names = tfidf_bigram.get_feature_names_out()
tfidf_df = pd.DataFrame(
    tfidf_bigram_matrix.toarray(),
    columns=feature_names
)

print("TF-IDF with Bigrams:")
print(tfidf_df)

# See how bigrams capture sentiment
print("\\nKey bigram features:")
print([f for f in feature_names if ' ' in f])  # Only bigrams
# Should include: 'not good', 'very good', 'not like', 'really liked'
\`\`\`

### N-gram from Scratch

\`\`\`python
def generate_ngrams (text, n):
    """Generate n-grams from text"""
    words = text.lower().split()
    ngrams = []
    
    for i in range (len (words) - n + 1):
        ngram = ' '.join (words[i:i+n])
        ngrams.append (ngram)
    
    return ngrams

# Examples
text = "machine learning is awesome"

print("Unigrams:", generate_ngrams (text, 1))
print("Bigrams:", generate_ngrams (text, 2))
print("Trigrams:", generate_ngrams (text, 3))

# Implement full n-gram vectorizer
class SimpleNgramVectorizer:
    """N-gram vectorizer from scratch"""
    
    def __init__(self, ngram_range=(1, 1)):
        self.ngram_range = ngram_range
        self.vocabulary = {}
    
    def fit (self, documents):
        """Build n-gram vocabulary"""
        all_ngrams = set()
        
        for doc in documents:
            # Generate n-grams for all n in range
            for n in range (self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams = generate_ngrams (doc, n)
                all_ngrams.update (ngrams)
        
        self.vocabulary = {ngram: idx for idx, ngram in enumerate (sorted (all_ngrams))}
        return self
    
    def transform (self, documents):
        """Transform documents to n-gram vectors"""
        vectors = []
        
        for doc in documents:
            vector = np.zeros (len (self.vocabulary))
            
            # Count all n-grams
            for n in range (self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams = generate_ngrams (doc, n)
                ngram_counts = Counter (ngrams)
                
                for ngram, count in ngram_counts.items():
                    if ngram in self.vocabulary:
                        idx = self.vocabulary[ngram]
                        vector[idx] = count
            
            vectors.append (vector)
        
        return np.array (vectors)
    
    def fit_transform (self, documents):
        self.fit (documents)
        return self.transform (documents)

# Test
ngram_vec = SimpleNgramVectorizer (ngram_range=(1, 2))
ngram_matrix = ngram_vec.fit_transform (documents)
print(f"\\nCustom n-gram vectorizer: {ngram_matrix.shape}")
print(f"Vocabulary size: {len (ngram_vec.vocabulary)}")
\`\`\`

## Practical Considerations

### Curse of Dimensionality

\`\`\`python
# Demonstrate vocabulary explosion with n-grams
sample_docs = [
    " ".join([f"word{i}" for i in range(100)])  # 100 unique words
] * 10

# Unigrams
vec_1 = CountVectorizer (ngram_range=(1, 1))
vec_1.fit (sample_docs)
print(f"Unigrams: {len (vec_1.vocabulary_)} features")

# + Bigrams  
vec_2 = CountVectorizer (ngram_range=(1, 2))
vec_2.fit (sample_docs)
print(f"Unigrams + Bigrams: {len (vec_2.vocabulary_)} features")

# + Trigrams
vec_3 = CountVectorizer (ngram_range=(1, 3))
vec_3.fit (sample_docs)
print(f"Up to Trigrams: {len (vec_3.vocabulary_)} features")

print("\\nFeatures grow exponentially with n-gram size!")
print("Solution: Use max_features or min_df to limit vocabulary")
\`\`\`

### Sparse Matrices

\`\`\`python
from scipy.sparse import csr_matrix

# BoW/TF-IDF create sparse matrices (most values are 0)
vec = TfidfVectorizer()
sparse_matrix = vec.fit_transform (documents)

print(f"Matrix shape: {sparse_matrix.shape}")
print(f"Matrix type: {type (sparse_matrix)}")
print(f"Sparsity: {(1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])) * 100:.2f}% zeros")

# Memory usage comparison
dense_matrix = sparse_matrix.toarray()
import sys
print(f"\\nSparse matrix size: {sparse_matrix.data.nbytes / 1024:.2f} KB")
print(f"Dense matrix size: {sys.getsizeof (dense_matrix) / 1024:.2f} KB")
print("Sparse representation saves memory!")
\`\`\`

### Complete Pipeline Example

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample sentiment data
texts = [
    "I love this product",
    "This is amazing",
    "Best purchase ever",
    "Terrible quality",
    "Waste of money",
    "Very disappointing"
] * 20

labels = [1, 1, 1, 0, 0, 0] * 20  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )),
    ('classifier', LogisticRegression (max_iter=200))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report (y_test, y_pred, target_names=['Negative', 'Positive']))

# Inspect important features
tfidf = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['classifier']

feature_names = tfidf.get_feature_names_out()
coefficients = classifier.coef_[0]

# Top positive features
top_positive = np.argsort (coefficients)[-10:]
print("\\nTop positive features (bigrams highlighted):")
for idx in top_positive[::-1]:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")

# Top negative features
top_negative = np.argsort (coefficients)[:10]
print("\\nTop negative features:")
for idx in top_negative:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")
\`\`\`

## Summary and Best Practices

### When to Use Each Method:

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **BoW** | Simple classification, topic modeling | Fast, simple, interpretable | Ignores word order, sparse |
| **TF-IDF** | Document similarity, search, classification | Weights importance, reduces common word impact | Still sparse, no semantics |
| **N-grams** | Sentiment, phrases, local context | Captures word order, phrases | Exponential growth, very sparse |

### Best Practices:

1. **Start simple**: Try BoW or TF-IDF before complex methods
2. **Limit vocabulary**: Use \`max_features\`, \`min_df\`, \`max_df\`
3. **Remove stopwords**: For most tasks (except sentiment with negations)
4. **Use bigrams selectively**: (1,2) range balances performance and dimensionality
5. **Keep it sparse**: Don't convert to dense unless necessary
6. **Normalize**: TF-IDF with L2 normalization works well for similarity tasks
7. **Feature selection**: Use techniques like chi-square to reduce dimensions
8. **Consider task**: Sentiment keeps more words/bigrams than topic modeling

### Limitations:

- **No semantic understanding**: "good" and "excellent" are different features
- **High dimensionality**: Can have 10,000+ features for large vocabularies
- **Sparsity**: Most values are zero, inefficient for neural networks
- **No word order**: "not good" vs "good not" are the same in BoW
- **OOV problem**: New words at inference time are ignored

**Solution Preview**: The next section covers word embeddings (Word2Vec, GloVe), which create dense, semantic representations that address many of these limitations.
`,
};
