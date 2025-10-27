import { QuizQuestion } from '../../../types';

export const textRepresentationQuiz: QuizQuestion[] = [
  {
    id: 'text-representation-dq-1',
    question:
      'Explain why TF-IDF typically performs better than simple Bag of Words for document similarity and search tasks. Use a concrete example to illustrate the key difference.',
    sampleAnswer: `TF-IDF outperforms Bag of Words for similarity and search tasks because it weights words by their discriminative power, not just frequency:

**The Key Problem with BoW:**

Bag of Words treats all words equally based solely on their frequency in a document. Common words that appear everywhere get high weights, drowning out rare, informative words.

**Concrete Example:**

Consider three documents:
1. "The machine learning algorithm performs classification tasks"
2. "The deep learning model performs image classification"  
3. "The weather forecast predicts rain"

**With Bag of Words:**

Word "the" appears 3 times across all documents
Word "performs" appears 2 times across documents
Word "algorithm" appears 1 time in one document

When representing document 1:
- "the": high count (appears twice)
- "algorithm": low count (appears once)
- Both weighted equally by frequency

When computing similarity between docs 1 and 2:
- High similarity because both contain "the", "performs", "classification"
- "the" contributes significantly to similarity despite being uninformative

When computing similarity between docs 1 and 3:
- Some similarity because both contain "the"
- "the" creates false similarity between unrelated documents

**With TF-IDF:**

IDF calculation:
- "the": log(3/3) = 0 (appears in all docs → zero weight!)
- "performs": log(3/2) = 0.18 (appears in 2 docs → low weight)
- "algorithm": log(3/1) = 0.48 (appears in 1 doc → high weight)
- "classification": log(3/2) = 0.18
- "forecast", "rain": log(3/1) = 0.48 (rare → high weight)

For document 1:
- "the": TF=2/7 × IDF=0 = 0 (correctly ignored!)
- "algorithm": TF=1/7 × IDF=0.48 = 0.07 (boosted!)
- "classification": TF=1/7 × IDF=0.18 = 0.03

Similarity between docs 1 and 2:
- High on meaningful words: "learning", "classification", "performs"
- Zero contribution from "the"
- More accurate similarity score

Similarity between docs 1 and 3:
- Low similarity (correctly)
- Common word "the" doesn't create false similarity
- Only meaningful words contribute

**Mathematical Intuition:**

TF (Term Frequency): "How important is this word in THIS document?"
IDF (Inverse Document Frequency): "How rare/specific is this word across ALL documents?"
TF-IDF = TF × IDF: "How important AND rare is this word?"

**The Discriminative Power:**

TF-IDF achieves better performance by:

1. **Down-weighting common words**: Words like "the", "is", "a" get near-zero IDF
2. **Up-weighting rare words**: Domain-specific terms get high IDF
3. **Context-aware**: A word's importance depends on both local and global frequency
4. **Better document vectors**: Vectors contain more signal, less noise

**Real-World Impact:**

Search engine example:
Query: "machine learning algorithms"

With BoW:
- Returns documents containing any high-frequency words
- May return documents about "the machine" or "learning English"
- Common words dominate ranking

With TF-IDF:
- Prioritizes documents where "machine", "learning", "algorithms" are rare/specific
- Filters out documents where these are common stopwords
- Returns more relevant results

**When BoW Might Be Better:**

- Very small vocabularies (IDF differences minimal)
- When all words are equally informative
- Specific tasks like author attribution (function words matter)
- When computational speed is critical (BoW is simpler)

**Practical Recommendation:**

Start with TF-IDF for:
- Document similarity/search
- Text classification
- Information retrieval
- Content recommendation

Use BoW only for:
- Baseline comparisons
- Very simple tasks
- When interpretability is crucial
- When you need maximum speed

**Modern Context:**

With neural networks and word embeddings, both BoW and TF-IDF are being replaced by dense representations. However, TF-IDF remains a strong baseline and is still used in production systems for its simplicity and interpretability.`,
    keyPoints: [
      'BoW weights words only by frequency, treating all words equally',
      'TF-IDF combines term frequency (local) with inverse document frequency (global)',
      'Common words get low IDF, rare informative words get high IDF',
      'TF-IDF eliminates noise from high-frequency uninformative words',
      'Better for similarity/search: matches on meaningful terms, not common words',
      'TF-IDF vectors have higher discriminative power for distinguishing documents',
    ],
  },
  {
    id: 'text-representation-dq-2',
    question:
      'You are building a sentiment analysis system for product reviews. Should you use unigrams only, or include bigrams? Explain your reasoning with examples of how bigrams can change sentiment detection.',
    sampleAnswer: `For sentiment analysis, including bigrams (1,2) n-gram range is strongly recommended because they capture negations, modifiers, and sentiment-bearing phrases that unigrams alone would miss or misinterpret:

**The Negation Problem:**

**Example 1: "not good"**

Unigrams only:
- Tokens: ["not", "good",]
- "good" has positive sentiment → model sees positive signal
- "not" might be a stopword → removed entirely
- Result: Positive sentiment (WRONG!)

Unigrams + Bigrams:
- Tokens: ["not", "good", "not good",]
- "not good" is a distinct feature with negative sentiment
- Model learns "not good" ≠ "good"
- Result: Negative sentiment (CORRECT!)

**Example 2: "not bad"**

Unigrams: ["not", "bad",] → negative sentiment (WRONG - double negative!)
Bigrams: ["not", "bad", "not bad",] → positive/neutral sentiment (CORRECT!)

**The Intensifier Problem:**

**Example 3: "very good"**

Unigrams: ["very", "good",]
- "very" is often a stopword or has neutral sentiment
- "good" has mild positive sentiment
- Combined signal: mildly positive

Bigrams: ["very", "good", "very good",]
- "very good" is a distinct feature with strong positive sentiment
- Model learns intensity: "very good" > "good"
- Result: Strongly positive sentiment (MORE ACCURATE!)

**Example 4: "absolutely terrible"**

Unigrams: ["absolutely", "terrible",]
- Misses the intensification effect

Bigrams: ["absolutely", "terrible", "absolutely terrible",]
- Captures extreme negative sentiment

**The Phrase Problem:**

**Example 5: "not worth the money"**

Unigrams: ["not", "worth", "money",]
- Individual words have mixed or neutral sentiment
- Hard to detect overall negative sentiment

Bigrams: ["not worth", "worth the", "the money",]
- "not worth" is a clear negative phrase
- Model can learn this specific phrase pattern

**Example 6: "waste of money"**

Unigrams: ["waste", "of", "money",]
- "waste" is negative, but "money" is neutral
- "of" is a stopword

Bigrams: ["waste of", "of money", "waste of money",] (with trigrams)
- "waste of money" is a strongly negative phrase in reviews
- Much clearer signal

**Practical Implementation:**

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample review data
reviews = [
    "This product is not good at all",
    "Very good quality",
    "Not bad for the price",
    "Absolutely terrible experience",
    "Good but not great",
    "Waste of money",
]

labels = [0, 1, 1, 0, 0, 0]  # 0=negative, 1=positive

# Unigrams only
pipeline_unigram = Pipeline([
    ('tfidf', TfidfVectorizer (ngram_range=(1,1))),
    ('clf', LogisticRegression())
])

# Unigrams + Bigrams
pipeline_bigram = Pipeline([
    ('tfidf', TfidfVectorizer (ngram_range=(1,2))),
    ('clf', LogisticRegression())
])

# The bigram model will learn:
# - "not good" → negative weight
# - "very good" → positive weight  
# - "not bad" → positive/neutral weight
# - "waste of" → negative weight
\`\`\`

**Trade-offs:**

**Pros of Including Bigrams:**
- Captures negations correctly
- Learns sentiment intensifiers
- Recognizes multi-word phrases
- Better accuracy (typically +3-7% F1)
- Handles contextual sentiment

**Cons of Including Bigrams:**
- Increased vocabulary size (2-5x larger)
- More sparse features
- Slower training/inference
- Risk of overfitting with small datasets
- Increased memory requirements

**Quantitative Impact:**

From empirical studies on sentiment analysis:
- Unigrams only: ~75-80% accuracy
- Unigrams + Bigrams: ~82-87% accuracy
- Unigrams + Bigrams + Trigrams: ~83-88% (diminishing returns)

The 5-7% improvement from bigrams is significant in production systems.

**Optimal Strategy:**

\`\`\`python
# Recommended configuration for sentiment analysis
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),  # Unigrams + bigrams
    max_features=10000,   # Limit to top 10k features
    min_df=2,             # Ignore very rare terms
    max_df=0.9,           # Ignore very common terms
    stop_words=None,      # DON'T remove stopwords (they matter for negation!)
    sublinear_tf=True     # Use log scaling
)
\`\`\`

**Important Note on Stopwords:**

DO NOT remove stopwords for sentiment analysis when using bigrams:
- "not" is a stopword but critical for "not good"
- "no" is a stopword but critical for "no problem"
- "very" is often removed but critical for "very bad"

**When to Skip Bigrams:**

- Very limited training data (<1000 samples): overfitting risk
- Real-time inference with strict latency requirements
- Memory-constrained environments
- Simple sentiment (no negations or modifiers)

**Modern Alternatives:**

With transformer models (BERT, GPT), n-grams become less critical because:
- Transformers learn context automatically
- Attention mechanism captures word relationships
- Subword tokenization handles rare phrases

However, for classical ML (Logistic Regression, Naive Bayes, SVM), bigrams remain essential for competitive sentiment analysis performance.

**Recommendation:**

For production sentiment analysis with classical ML:
- **Always use (1,2) n-gram range minimum**
- Consider (1,3) for review-heavy phrases
- Keep stopwords
- Limit features with max_features to control dimensionality
- Monitor performance improvement to justify complexity`,
    keyPoints: [
      'Negations reverse sentiment: "not good" must be captured as a unit',
      'Intensifiers modify sentiment: "very good" is stronger than "good"',
      'Multi-word phrases have distinct sentiment: "waste of money"',
      'Bigrams typically improve sentiment accuracy by 5-7%',
      'Trade-off: improved accuracy vs increased dimensionality and sparsity',
      'Do NOT remove stopwords when using bigrams for sentiment analysis',
      'Bigrams less critical with transformers but essential for classical ML',
    ],
  },
  {
    id: 'text-representation-dq-3',
    question:
      'Explain the "curse of dimensionality" problem with count-based text representations (BoW/TF-IDF with n-grams), and describe practical strategies to mitigate it in production systems.',
    sampleAnswer: `The curse of dimensionality refers to the exponential growth of feature space as vocabulary size increases, especially with n-grams, leading to computational, statistical, and practical problems:

**The Dimensionality Problem:**

**Example: Vocabulary Explosion**

Small corpus with 1,000 unique words:
- Unigrams only: 1,000 features
- Add bigrams: ~1,000² = ~1,000,000 potential bigram features
- Add trigrams: ~1,000³ = ~1,000,000,000 potential trigram features!

Real corpus with 50,000 unique words:
- Unigrams: 50,000 features
- Unigrams + bigrams: 50,000 + 50,000² = ~2.5 billion features!
- Even filtering reduces this to millions of features

**Why This Is Problematic:**

**1. Memory Issues:**

\`\`\`python
# Example calculation
vocab_size = 100,000  # 100k unique features
num_documents = 10,000

# Dense matrix memory
dense_memory = vocab_size * num_documents * 8 bytes  # float64
print(f"Dense matrix: {dense_memory / 1e9:.2f} GB")  # 8 GB!

# Even sparse matrices with 0.1% density
sparse_density = 0.001
sparse_memory = vocab_size * num_documents * sparse_density * 8 bytes
print(f"Sparse matrix: {sparse_memory / 1e6:.2f} MB")  # 800 MB
\`\`\`

**2. Computational Cost:**

Training time scales with dimensionality:
- Logistic Regression: O(n_features × n_samples × n_iterations)
- SVM: O(n_features × n_samples²)
- Random Forest: O(n_trees × n_features × n_samples × log (n_samples))

With millions of features, training becomes prohibitively slow.

**3. Sparsity:**

\`\`\`
Most documents use <1% of vocabulary
Each document vector has 99%+ zeros
Extreme sparsity degrades model performance
\`\`\`

**4. Overfitting:**

Many features relative to samples → model memorizes training data
Rare n-grams appear in few documents → unreliable statistics
Model learns noise instead of signal

**5. Generalization:**

Test documents contain unseen n-grams (OOV)
Model cannot utilize huge portion of learned features
Poor generalization to new data

**Practical Mitigation Strategies:**

**Strategy 1: Vocabulary Filtering**

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,    # Keep only top 10k features
    min_df=5,               # Ignore terms appearing in <5 documents
    max_df=0.7,             # Ignore terms in >70% of documents
    ngram_range=(1, 2)      # Limit to bigrams
)

# Rationale:
# - max_features: Limits absolute size
# - min_df: Removes rare/noisy terms (likely typos or very specific)
# - max_df: Removes overly common terms (little discriminative power)
\`\`\`

**Effect:**
- Reduces features by 90-99%
- Typically loses <5% accuracy
- Massive computational savings

**Strategy 2: Feature Selection**

\`\`\`python
from sklearn.feature_selection import SelectKBest, chi2

# Chi-square feature selection
selector = SelectKBest (chi2, k=5000)

X_selected = selector.fit_transform(X_train, y_train)

# Keeps only top 5,000 features most correlated with labels
# Removes noise and irrelevant features
\`\`\`

**Other selection methods:**
- Mutual information
- L1 regularization (Lasso) for embedded selection
- Tree-based feature importance

**Strategy 3: Dimensionality Reduction**

\`\`\`python
from sklearn.decomposition import TruncatedSVD

# Reduce 100,000 features to 300 dense dimensions
svd = TruncatedSVD(n_components=300)
X_reduced = svd.fit_transform(X_sparse)

# Latent Semantic Analysis (LSA) for text
# Projects sparse high-dim vectors to dense low-dim space
\`\`\`

**Benefits:**
- Dense representations (no sparsity)
- Captures latent semantic relationships
- Much faster for downstream models

**Strategy 4: Subword Tokenization**

\`\`\`python
# Instead of word-level n-grams, use character n-grams
vectorizer_char = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),  # 3-5 character sequences
    max_features=10000
)

# Or use subword tokenization (BPE)
# Handles typos, morphology, unseen words
# More robust vocabulary with controlled size
\`\`\`

**Strategy 5: Hashing Trick**

\`\`\`python
from sklearn.feature_extraction.text import HashingVectorizer

# Hash features to fixed-size space
vectorizer_hash = HashingVectorizer (n_features=2**16)  # 65k features

# No need to store vocabulary
# Fixed memory footprint
# Handles new words automatically
# (Small collision risk)
\`\`\`

**Strategy 6: Careful N-gram Selection**

\`\`\`python
# Instead of all bigrams, use only those with stopwords
# "not good", "very bad", etc. - sentiment-critical
vectorizer_selective = TfidfVectorizer(
    ngram_range=(1, 2),
    token_pattern=r'\\b\\w+\\b',  # Custom pattern
    # Custom tokenizer that keeps only bigrams with specific patterns
)

# Manually filter n-grams after vectorization
# Keep only n-grams above certain TF-IDF threshold
\`\`\`

**Strategy 7: Pipeline Optimization**

\`\`\`python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8
    )),
    ('svd', TruncatedSVD(n_components=300)),  # Reduce dimensions
    ('clf', LogisticRegression())
])

# TF-IDF → SVD → Classifier
# Combines multiple strategies
\`\`\`

**Strategy 8: Monitoring and Iteration**

\`\`\`python
# Track vocabulary growth
vocab_sizes = []
accuracies = []

for max_feat in [1000, 5000, 10000, 50000, 100000]:
    vec = TfidfVectorizer (max_features=max_feat)
    X = vec.fit_transform (docs)
    
    vocab_sizes.append (len (vec.vocabulary_))
    # Train model and measure accuracy
    # Plot accuracy vs vocabulary size
    
# Find elbow point: diminishing returns beyond certain size
\`\`\`

**Decision Framework:**

| Corpus Size | Vocabulary | Strategy |
|-------------|------------|----------|
| <10k docs | <50k features | max_features=10k, min_df=2 |
| 10k-100k docs | <200k features | max_features=50k, min_df=5, feature selection |
| >100k docs | >500k features | max_features=100k, min_df=10, SVD reduction |

**Production Best Practices:**1. **Start conservative**: Begin with max_features=10000
2. **Measure impact**: Increase gradually, measure accuracy gains
3. **Monitor memory**: Track RAM usage in production
4. **Use sparse**: Keep data in sparse format until necessary
5. **Feature engineering over volume**: Better features > more features
6. **Consider alternatives**: Embeddings (Word2Vec, BERT) for dense representations

**Modern Solution:**

Dense embeddings (Word2Vec, GloVe, BERT) largely solve dimensionality issues:
- Fixed, small dimensionality (100-768 dims)
- Dense vectors (no sparsity)
- Semantic meaning captured
- Better generalization

However, TF-IDF with proper dimensionality control remains a strong, interpretable baseline.

**Key Insight:**

The curse of dimensionality is manageable with proper feature engineering and selection. The goal is not to use all possible features, but to use the RIGHT features—those that maximize signal while minimizing noise and computational cost.`,
    keyPoints: [
      'N-grams cause exponential vocabulary growth: 1000 words → 1M+ bigrams',
      'High dimensionality leads to sparsity, overfitting, memory issues, slow training',
      'Mitigation: max_features, min_df/max_df filtering reduces features by 90%+',
      'Feature selection (chi-square, L1) keeps only discriminative features',
      'Dimensionality reduction (SVD/LSA) creates dense representations',
      'Modern solution: word embeddings provide fixed-size dense representations',
      'Trade-off: reducing dimensions may lose <5% accuracy but gain 10-100x speed',
    ],
  },
];
