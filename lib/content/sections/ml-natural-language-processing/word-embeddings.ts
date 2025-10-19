/**
 * Section: Word Embeddings
 * Module: Natural Language Processing
 *
 * Covers Word2Vec (Skip-gram, CBOW), GloVe, FastText, and distributed word representations
 */

export const wordEmbeddingsSection = {
  id: 'word-embeddings',
  title: 'Word Embeddings',
  content: `
# Word Embeddings

## Introduction

Word embeddings revolutionized NLP by representing words as dense, low-dimensional vectors that capture semantic meaning. Unlike sparse one-hot or count-based representations, embeddings place similar words close together in vector space, enabling models to understand that "king" is similar to "queen" and different from "banana."

**Key Breakthrough:**
- **Dense representations**: 100-300 dimensions instead of 10,000+ sparse dimensions
- **Semantic similarity**: Vector distance reflects meaning similarity
- **Learned from data**: Automatically discover word relationships
- **Transfer learning**: Pre-trained embeddings work across tasks

**Famous Example:**
\`\`\`
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
\`\`\`

This section covers Word2Vec, GloVe, and FastText—foundational embedding methods still widely used today.

## The Distributional Hypothesis

**Core Idea**: "You shall know a word by the company it keeps" - J.R. Firth

Words appearing in similar contexts have similar meanings.

\`\`\`python
# Sentence 1: "The cat sat on the mat"
# Sentence 2: "The dog sat on the log"
# Context around "cat" and "dog" is similar → they should have similar embeddings

# Sentence 3: "The car drove down the street"
# Context around "car" is different → different embedding
\`\`\`

**Intuition:**
- Words with similar meanings appear in similar contexts
- We can learn word meanings from large text corpora
- No manual labeling required—fully unsupervised!

## Word2Vec: Skip-gram and CBOW

Word2Vec (Mikolov et al., 2013) popularized word embeddings with two architectures:
- **Skip-gram**: Predict context words from target word
- **CBOW**: Predict target word from context words

### Skip-gram Architecture

Given a target word, predict its surrounding context words.

\`\`\`python
# Sentence: "I love machine learning"
# Window size: 2

# Target: "machine"
# Context: ["I", "love", "learning"]

# Training pairs (target → context):
# machine → I
# machine → love  
# machine → learning
\`\`\`

**Architecture:**
\`\`\`
Input (one-hot encoded target word) 
    ↓
Embedding Layer (weights = word vectors)
    ↓
Output Layer (softmax over vocabulary)
    ↓
Probability distribution over context words
\`\`\`

### CBOW Architecture

Given context words, predict the target word (opposite of skip-gram).

\`\`\`python
# Sentence: "I love machine learning"
# Window size: 2

# Context: ["I", "love", "learning"]
# Target: "machine"

# Average/sum context embeddings → predict target
\`\`\`

**Architecture:**
\`\`\`
Inputs (one-hot encoded context words)
    ↓
Embedding Layer (weights = word vectors)
    ↓
Average/Sum context embeddings
    ↓
Output Layer (softmax over vocabulary)
    ↓
Probability of target word
\`\`\`

### Training Word2Vec from Scratch

\`\`\`python
import numpy as np
from collections import defaultdict
import re

class SimpleWord2Vec:
    """Simplified Word2Vec Skip-gram implementation"""
    
    def __init__(self, embedding_dim=100, window_size=2, learning_rate=0.01):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.vocab = {}
        self.inverse_vocab = {}
        
    def build_vocab(self, sentences):
        """Build vocabulary from sentences"""
        word_counts = defaultdict(int)
        
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                word_counts[word] += 1
        
        # Create vocabulary (most common words)
        self.vocab = {word: idx for idx, (word, count) in 
                      enumerate(sorted(word_counts.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True))}
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def generate_training_data(self, sentences):
        """Generate (target, context) pairs"""
        training_data = []
        
        for sentence in sentences:
            words = sentence.lower().split()
            
            for idx, target_word in enumerate(words):
                if target_word not in self.vocab:
                    continue
                    
                target_idx = self.vocab[target_word]
                
                # Get context words (within window)
                start = max(0, idx - self.window_size)
                end = min(len(words), idx + self.window_size + 1)
                
                for context_idx in range(start, end):
                    if context_idx != idx:
                        context_word = words[context_idx]
                        if context_word in self.vocab:
                            context_word_idx = self.vocab[context_word]
                            training_data.append((target_idx, context_word_idx))
        
        return training_data
    
    def initialize_embeddings(self):
        """Initialize embedding matrices"""
        # Input embeddings (target words)
        self.W_in = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        
        # Output embeddings (context words)
        self.W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def train(self, sentences, epochs=5):
        """Train Skip-gram model"""
        self.build_vocab(sentences)
        training_data = self.generate_training_data(sentences)
        self.initialize_embeddings()
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Training pairs: {len(training_data)}")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for target_idx, context_idx in training_data:
                # Forward pass
                h = self.W_in[target_idx]  # Hidden layer (embedding)
                u = np.dot(h, self.W_out)  # Output scores
                y_pred = self.softmax(u)   # Probabilities
                
                # Compute loss
                loss = -np.log(y_pred[context_idx] + 1e-10)
                total_loss += loss
                
                # Backpropagation
                # Gradient of output layer
                y_pred[context_idx] -= 1  # y_pred - y_true
                dW_out = np.outer(h, y_pred)
                
                # Gradient of input layer
                dh = np.dot(y_pred, self.W_out.T)
                
                # Update weights
                self.W_out -= self.learning_rate * dW_out.T
                self.W_in[target_idx] -= self.learning_rate * dh
            
            if (epoch + 1) % 1 == 0:
                avg_loss = total_loss / len(training_data)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_embedding(self, word):
        """Get embedding vector for a word"""
        if word in self.vocab:
            return self.W_in[self.vocab[word]]
        return None
    
    def most_similar(self, word, top_n=5):
        """Find most similar words using cosine similarity"""
        if word not in self.vocab:
            return []
        
        word_vec = self.get_embedding(word)
        
        # Compute cosine similarity with all words
        similarities = []
        for other_word, idx in self.vocab.items():
            if other_word == word:
                continue
            
            other_vec = self.W_in[idx]
            
            # Cosine similarity
            similarity = np.dot(word_vec, other_vec) / (
                np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-10
            )
            similarities.append((other_word, similarity))
        
        # Return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

# Example usage
sentences = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are animals",
    "the quick brown fox jumps",
    "machine learning is fascinating",
    "deep learning uses neural networks",
    "natural language processing is important",
] * 100  # Repeat for more training data

# Train
w2v = SimpleWord2Vec(embedding_dim=50, window_size=2, learning_rate=0.01)
w2v.train(sentences, epochs=10)

# Test similarity
print("\\nMost similar to 'cat':")
print(w2v.most_similar('cat', top_n=5))

print("\\nMost similar to 'learning':")
print(w2v.most_similar('learning', top_n=5))
\`\`\`

### Using Gensim for Word2Vec

Gensim provides an optimized, production-ready Word2Vec implementation.

\`\`\`python
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Sample corpus
sentences = [
    ['machine', 'learning', 'is', 'fascinating'],
    ['deep', 'learning', 'uses', 'neural', 'networks'],
    ['natural', 'language', 'processing', 'is', 'important'],
    ['word', 'embeddings', 'capture', 'semantic', 'meaning'],
    ['transformers', 'revolutionized', 'natural', 'language', 'processing'],
    ['neural', 'networks', 'learn', 'representations'],
] * 100

# Train Word2Vec Skip-gram
model_sg = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window size
    min_count=1,          # Ignore words with frequency < min_count
    workers=4,            # Parallel threads
    sg=1,                 # 1 = skip-gram, 0 = CBOW
    epochs=10
)

# Train Word2Vec CBOW
model_cbow = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,                 # CBOW
    epochs=10
)

# Get word vector
vector = model_sg.wv['learning']
print(f"Vector for 'learning': {vector[:5]}... (shape: {vector.shape})")

# Find similar words
similar = model_sg.wv.most_similar('learning', topn=5)
print(f"\\nMost similar to 'learning':")
for word, score in similar:
    print(f"  {word}: {score:.4f}")

# Word analogy: king - man + woman ≈ queen
result = model_sg.wv.most_similar(positive=['natural', 'processing'], 
                                   negative=['language'], 
                                   topn=3)
print(f"\\nAnalogy (natural + processing - language):")
for word, score in result:
    print(f"  {word}: {score:.4f}")

# Cosine similarity between two words
similarity = model_sg.wv.similarity('learning', 'networks')
print(f"\\nSimilarity between 'learning' and 'networks': {similarity:.4f}")

# Check if word exists
print(f"\\n'learning' in vocabulary: {'learning' in model_sg.wv}")
print(f"'unknown_word' in vocabulary: {'unknown_word' in model_sg.wv}")
\`\`\`

### Training on Large Corpus

\`\`\`python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# For large text files (one sentence per line)
# LineSentence efficiently streams data without loading all into memory

# Train on large corpus
model = Word2Vec(
    sentences=LineSentence('large_corpus.txt'),  # Stream sentences
    vector_size=300,
    window=5,
    min_count=5,          # Ignore rare words
    workers=8,
    sg=1,
    negative=5,           # Negative sampling
    ns_exponent=0.75,
    epochs=5,
    callbacks=[],         # Add callbacks for monitoring
)

# Save model
model.save('word2vec_model.bin')

# Load model
loaded_model = Word2Vec.load('word2vec_model.bin')

# Save just the word vectors (more compact)
model.wv.save('word2vec_vectors.kv')

# Load just vectors
from gensim.models import KeyedVectors
wv = KeyedVectors.load('word2vec_vectors.kv')
\`\`\`

## GloVe (Global Vectors)

GloVe (Pennington et al., 2014) learns embeddings by factorizing word co-occurrence matrices. Unlike Word2Vec's local context windows, GloVe uses global corpus statistics.

**Key Idea:**
- Construct word co-occurrence matrix from entire corpus
- Factorize matrix to get word vectors
- Vectors that predict co-occurrence well

### Using Pre-trained GloVe

\`\`\`python
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Download GloVe from: https://nlp.stanford.edu/projects/glove/
# Convert GloVe format to Word2Vec format
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Load GloVe vectors
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Use like Word2Vec
vector = glove_model['computer']
similar = glove_model.most_similar('computer', topn=5)

print("Most similar to 'computer':")
for word, score in similar:
    print(f"  {word}: {score:.4f}")

# Famous analogy: king - man + woman ≈ queen
result = glove_model.most_similar(positive=['king', 'woman'], 
                                   negative=['man'], 
                                   topn=1)
print(f"\\nking - man + woman ≈ {result[0][0]} (score: {result[0][1]:.4f})")
\`\`\`

## FastText

FastText (Bojanowski et al., 2017) extends Word2Vec by representing words as bags of character n-grams. This handles out-of-vocabulary words and morphological variations.

**Key Innovation:**
- Word represented as sum of character n-gram embeddings
- "learning" = <le + lea + ear + arn + rni + nin + ing + ng>
- Can generate embeddings for unseen words!

### Training FastText

\`\`\`python
from gensim.models import FastText

sentences = [
    ['machine', 'learning', 'is', 'fascinating'],
    ['deep', 'learning', 'uses', 'neural', 'networks'],
    ['unsupervised', 'learning', 'discovers', 'patterns'],
    ['reinforcement', 'learning', 'learns', 'actions'],
] * 100

# Train FastText
model_ft = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,                 # Skip-gram
    min_n=3,              # Min character n-gram length
    max_n=6,              # Max character n-gram length
    epochs=10
)

# Get embedding for known word
vector_known = model_ft.wv['learning']
print(f"Vector for 'learning': {vector_known[:5]}...")

# Get embedding for OOV word (!)
# FastText can generate embeddings for unseen words using character n-grams
vector_oov = model_ft.wv['learnings']  # Not in training!
print(f"\\nVector for 'learnings' (OOV): {vector_oov[:5]}...")

# Similar words to OOV
similar_oov = model_ft.wv.most_similar('learnings', topn=5)
print(f"\\nMost similar to 'learnings' (OOV):")
for word, score in similar_oov:
    print(f"  {word}: {score:.4f}")

# Handle typos
vector_typo = model_ft.wv['lerning']  # Typo!
similar_typo = model_ft.wv.most_similar('lerning', topn=3)
print(f"\\nMost similar to 'lerning' (typo):")
for word, score in similar_typo:
    print(f"  {word}: {score:.4f}")
\`\`\`

### Using Pre-trained FastText

\`\`\`python
import gensim.downloader as api

# Download pre-trained FastText (warning: large file!)
# fasttext_model = api.load('fasttext-wiki-news-subwords-300')

# Or load from file
from gensim.models import FastText
fasttext_model = FastText.load('fasttext_model.bin')

# Use for OOV handling
embedding = fasttext_model.wv['unbelievable']  # Even if not in training vocab
\`\`\`

## Comparing Word2Vec, GloVe, and FastText

\`\`\`python
from gensim.models import Word2Vec, FastText, KeyedVectors

# Train all three (using same corpus for fair comparison)
sentences = [['machine', 'learning']] * 1000  # Simplified

# Word2Vec
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=10)

# FastText
ft = FastText(sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=10)

# GloVe (load pre-trained for comparison)
# glove = KeyedVectors.load_word2vec_format('glove.txt')

print("Comparison:")
print("=" * 60)

# Known word
word = 'learning'
print(f"\\n1. Known word: '{word}'")
print(f"   Word2Vec: ✓ ({w2v.wv[word][:3]}...)")
print(f"   FastText: ✓ ({ft.wv[word][:3]}...)")

# OOV word
oov_word = 'learnings'
print(f"\\n2. OOV word: '{oov_word}'")
try:
    vec = w2v.wv[oov_word]
    print(f"   Word2Vec: ✓ (found)")
except KeyError:
    print(f"   Word2Vec: ✗ (KeyError - word not in vocabulary)")

print(f"   FastText: ✓ ({ft.wv[oov_word][:3]}...) <- Can handle OOV!")

# Morphological variations
print(f"\\n3. Morphological similarity:")
words = ['learn', 'learning', 'learned', 'learner']
for w1 in words:
    for w2 in words:
        if w1 != w2:
            try:
                sim_ft = ft.wv.similarity(w1, w2)
                print(f"   FastText: {w1} <-> {w2}: {sim_ft:.4f}")
            except:
                pass
    break
\`\`\`

### Summary Comparison

| Feature | Word2Vec | GloVe | FastText |
|---------|----------|-------|----------|
| **Training** | Local context windows | Global co-occurrence | Local context + char n-grams |
| **Speed** | Fast | Medium | Slower (n-grams) |
| **OOV handling** | ✗ No | ✗ No | ✓ Yes |
| **Morphology** | ✗ No | ✗ No | ✓ Yes |
| **Memory** | Low | Medium | Higher (n-grams) |
| **Quality** | Good | Good | Good + OOV |
| **Best for** | General | Similarity tasks | Rare words, typos |

## Practical Applications

### Document Similarity

\`\`\`python
import numpy as np
from gensim.models import Word2Vec

# Train model
sentences = [['machine', 'learning'], ['deep', 'learning']] * 100
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

def document_vector(doc, model):
    """Compute document embedding by averaging word embeddings"""
    vectors = []
    for word in doc.split():
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

# Documents
doc1 = "machine learning is amazing"
doc2 = "deep learning uses neural networks"
doc3 = "the cat sat on the mat"

# Get document vectors
vec1 = document_vector(doc1, model)
vec2 = document_vector(doc2, model)
vec3 = document_vector(doc3, model)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity

sim_12 = cosine_similarity([vec1], [vec2])[0][0]
sim_13 = cosine_similarity([vec1], [vec3])[0][0]

print(f"Similarity (doc1, doc2): {sim_12:.4f}")  # High (both about ML)
print(f"Similarity (doc1, doc3): {sim_13:.4f}")  # Low (different topics)
\`\`\`

### Text Classification with Embeddings

\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data
texts = [
    "I love this movie",
    "Great film",
    "Terrible movie",
    "Awful experience"
] * 100

labels = [1, 1, 0, 0] * 100  # 1=positive, 0=negative

# Convert texts to vectors using word embeddings
X = np.array([document_vector(text, model) for text in texts])
y = np.array(labels)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Classification accuracy: {accuracy:.2f}")
\`\`\`

## Evaluation and Visualization

\`\`\`python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get embeddings for a set of words
words = ['king', 'queen', 'man', 'woman', 'learning', 'teaching', 
         'cat', 'dog', 'computer', 'algorithm']

# Filter words that exist in vocabulary
words_in_vocab = [w for w in words if w in model.wv]
vectors = np.array([model.wv[w] for w in words_in_vocab])

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

for i, word in enumerate(words_in_vocab):
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))

plt.title("Word Embeddings Visualization (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
\`\`\`

## Best Practices

1. **Choose the right model:**
   - Word2Vec: Fast, general purpose
   - GloVe: Good for similarity tasks
   - FastText: Best for rare words, typos, morphology

2. **Hyperparameters:**
   - vector_size: 100-300 for most tasks
   - window: 5-10 for general text
   - min_count: Filter rare words (5-10 for large corpora)
   - epochs: 5-20 depending on corpus size

3. **Use pre-trained embeddings:**
   - Train on billions of tokens: better quality
   - Transfer learning: works well across domains
   - Save training time

4. **Handle OOV:**
   - Use FastText for automatic OOV handling
   - Or use <UNK> token with Word2Vec
   - Or average embeddings of similar words

5. **Evaluation:**
   - Intrinsic: Word similarity, analogies
   - Extrinsic: Downstream task performance
   - Always evaluate on your specific task

## Limitations and Future Directions

**Limitations:**
- **Single vector per word**: "bank" (financial) and "bank" (river) have same embedding
- **Static**: Doesn't adapt to document context
- **Training data bias**: Embeddings reflect biases in training corpus

**Solutions (covered in next sections):**
- **Contextualized embeddings**: ELMo, BERT give different vectors for same word in different contexts
- **Subword embeddings**: Better handling of morphology
- **Debiasing techniques**: Remove demographic biases

**Key Takeaway:**
Word embeddings transformed NLP by learning semantic representations from data. They remain foundational, even as contextualized embeddings (BERT, GPT) become dominant.
`,
};
