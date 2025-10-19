import { QuizQuestion } from '../../../types';

export const textPreprocessingQuiz: QuizQuestion[] = [
  {
    id: 'text-preprocessing-dq-1',
    question:
      'Explain why sentiment analysis tasks often benefit from MINIMAL text preprocessing, while topic modeling benefits from AGGRESSIVE preprocessing. What specific preprocessing steps should differ and why?',
    sampleAnswer: `The preprocessing strategy should match the task's requirements for understanding meaning:

**Sentiment Analysis - Minimal Preprocessing:**

Sentiment analysis aims to detect emotional tone and opinions, which are heavily influenced by:

1. **Punctuation**: Multiple exclamation marks (!!! vs !) indicate intensity
   - "I liked it." vs "I LOVED it!!!" - very different sentiments
   - Question marks can indicate uncertainty or sarcasm

2. **Capitalization**: Emphasis and shouting
   - "This is good" vs "This is AMAZING" - case conveys emotion
   - All-caps often indicates strong feelings

3. **Negation words**: Critical for meaning
   - "not good" vs "good" - stopword removal would destroy meaning
   - "not bad" vs "bad" - negation flips sentiment

4. **Word forms**: Tense and aspect matter
   - "I loved it" (past satisfaction) vs "I love it" (present)
   - "will buy again" vs "would never buy" - tense indicates intent

**Recommended for Sentiment:**
- Keep: punctuation, capitalization, stopwords, tense
- Remove: HTML tags, URLs (noise but not signal)
- Minimal normalization: preserve emotional markers

**Topic Modeling - Aggressive Preprocessing:**

Topic modeling (LDA, NMF) aims to discover thematic structure across documents, focusing on content words:

1. **Remove stopwords**: Eliminate high-frequency, low-information words
   - "the", "is", "at" appear in all topics, add no discrimination
   - Reduces noise and improves topic coherence

2. **Lemmatization**: Group semantically related words
   - "running", "runs", "ran" → "run" (all same concept)
   - Reduces vocabulary size and improves pattern detection

3. **Remove punctuation**: Not relevant for topic discovery
   - Commas and periods don't define topics
   - Clean tokens for cleaner term-document matrices

4. **Lowercase everything**: Case doesn't affect topics
   - "Machine Learning" and "machine learning" are the same topic
   - Reduces vocabulary duplication

5. **Filter short tokens**: "a", "I", "it" rarely indicate topics
   - Minimum length 3-4 characters focuses on content words

**Recommended for Topics:**
- Remove: stopwords, punctuation, numbers, short tokens
- Apply: lemmatization, lowercasing
- Goal: extract core semantic content only

**The Fundamental Difference:**

- **Sentiment**: How something is said matters (style, emphasis, emotion)
- **Topics**: What is being discussed matters (content, themes, subjects)

**Practical Example:**

Text: "I absolutely LOVED this movie!!! The acting was incredible."

Sentiment preprocessing:
→ "I absolutely LOVED this movie !!! The acting was incredible ."
(Keeps emphasis and emotion markers)

Topic preprocessing:
→ "absolutely love movie acting incredible"
(Extracts core concepts only)

**Modern Context:**

With transformer models (BERT, GPT), even sentiment analysis benefits from minimal preprocessing since these models learn context-dependent representations that capture nuance from the full input. However, for classical ML (Naive Bayes, Logistic Regression), these preprocessing differences remain critical.`,
    keyPoints: [
      'Sentiment analysis needs emotional markers: punctuation, caps, negations preserved',
      'Topic modeling removes noise to focus on core content words',
      'Stopword removal helps topics but destroys sentiment (negations)',
      'Lemmatization groups concepts for topics but loses tense information',
      'How vs what: sentiment cares about expression style, topics about content',
      'Transformer models can learn from richer input; classical ML needs more preprocessing',
    ],
  },
  {
    id: 'text-preprocessing-dq-2',
    question:
      'A financial sentiment analysis system trained on news articles is performing poorly on social media posts (tweets). Describe what preprocessing inconsistencies might cause this, and propose a solution for handling both data sources.',
    sampleAnswer: `This is a classic domain mismatch problem where preprocessing designed for one text type fails on another:

**Preprocessing Inconsistencies:**

1. **Language Style Differences:**

News articles:
- Formal language: "The Federal Reserve announced..."
- Complete sentences with proper grammar
- Standard punctuation usage
- No slang or abbreviations

Twitter/Social:
- Informal: "Fed just dropped rates 🚀"
- Fragments, incomplete sentences
- Heavy use of emojis, hashtags, mentions
- Slang, abbreviations (BTW, IMHO, gonna)

2. **Specific Problems:**

**Hashtags:** Training preprocessor might remove hashtags as "noise", but they're critical signals
- \`#bullish\` \`#bearish\` carry explicit sentiment
- \`#FOMO\` \`#hodl\` are domain-specific sentiment indicators
- Solution: Preserve hashtags or extract them as features

**Emojis:** Training data likely has few emojis
- 🚀 🌕 (moon) = very bullish in crypto/stocks
- 📉 😱 = bearish sentiment
- Solution: Either preserve emojis or map to sentiment tokens

**@mentions:** News has "John Smith" vs Twitter has "@JohnSmith"
- Training preprocessor might remove @mentions as usernames
- Solution: Replace with <USER> token or keep for context

**URLs:** Both have URLs but different formats
- News: full URLs with context
- Twitter: t.co shortened links
- Solution: Consistent URL handling

**Character limits:** Twitter encourages abbreviations
- "going to" vs "gonna"
- "you" vs "u"
- Solution: Expand contractions and common abbreviations

3. **Vocabulary Mismatch:**

Training vocabulary:
- Formal: "declined", "announced", "reported"

Social media vocabulary:
- Informal: "crashed", "dumped", "mooning", "rekt"

This causes out-of-vocabulary (OOV) issues if vocabulary was frozen from training.

**Proposed Solution: Unified Preprocessing Pipeline**

\`\`\`python
class AdaptivePreprocessor:
    def __init__(self):
        # Emoji sentiment mapping
        self.emoji_sentiments = {
            '🚀': ' <POSITIVE> ',
            '🌕': ' <POSITIVE> ',
            '📈': ' <POSITIVE> ',
            '📉': ' <NEGATIVE> ',
            '😱': ' <NEGATIVE> ',
            '💎': ' <POSITIVE> ',  # diamond hands
            # ... more mappings
        }
        
        # Common abbreviation expansions
        self.social_expansions = {
            'btw': 'by the way',
            'imho': 'in my humble opinion',
            'gonna': 'going to',
            'wanna': 'want to',
            'lol': '<POSITIVE>',
            'smh': '<NEGATIVE>',
            # ... more expansions
        }
    
    def preprocess(self, text, source='news'):
        if source == 'social':
            # Handle emojis
            for emoji, sentiment in self.emoji_sentiments.items():
                text = text.replace(emoji, sentiment)
            
            # Expand hashtags
            text = re.sub(r'#(\\w+)', r'hashtag_\\1', text)
            
            # Replace mentions
            text = re.sub(r'@\\w+', '<USER>', text)
            
            # Expand abbreviations
            words = text.split()
            expanded = []
            for word in words:
                expanded.append(
                    self.social_expansions.get(word.lower(), word)
                )
            text = ' '.join(expanded)
        
        # Common preprocessing for both
        text = text.lower()
        text = self.clean_urls(text)
        text = self.normalize_whitespace(text)
        
        return text
\`\`\`

**Alternative Solution: Multi-Domain Training**

Instead of fixing preprocessing, retrain with both domains:

1. **Collect social media training data:**
   - Annotate tweets/posts with sentiment
   - Include emoji-heavy, informal examples
   - Balance with existing news data

2. **Domain adaptation techniques:**
   - Domain-adversarial training: model learns domain-invariant features
   - Transfer learning: fine-tune news model on social data
   - Multi-task learning: predict sentiment AND domain

3. **Use domain-robust models:**
   - Transformer models (BERT) handle varied text better
   - Character-level models adapt to spelling variations
   - Subword tokenization (BPE) handles unknown words

**Implementation Strategy:**

Phase 1: Quick fix
- Adapt preprocessing to handle social media features
- Map social slang to sentiment tokens
- Retrain on expanded vocabulary

Phase 2: Long-term solution
- Collect and annotate social media training data
- Use domain adaptation or multi-task learning
- Deploy domain-aware model that handles both sources

Phase 3: Monitoring
- Track performance by source domain
- Continuously collect social media data
- Regular retraining to adapt to evolving language

**Key Insight:**

The root cause is training on one domain (formal news) and deploying on another (informal social). The best solution combines:
1. Preprocessing that handles both domains
2. Training data from both domains
3. Models robust to domain shift (transformers)
4. Continuous monitoring and adaptation

This is common in production NLP systems where text sources vary widely.`,
    keyPoints: [
      'Domain mismatch: formal news vs informal social media language',
      'Social media specific: emojis, hashtags, mentions, slang, abbreviations',
      'Preprocessing must handle domain-specific features consistently',
      'Map emojis/slang to sentiment tokens rather than removing',
      'Long-term solution: train on both domains with domain adaptation',
      'Transformers and subword tokenization help with domain robustness',
    ],
  },
  {
    id: 'text-preprocessing-dq-3',
    question:
      'You are building an NLP pipeline that will be maintained for years. Explain why preprocessing consistency between training and inference is critical, and describe a production-grade approach to ensure consistency, including versioning and testing strategies.',
    sampleAnswer: `Preprocessing consistency is one of the most common sources of bugs in production NLP systems, often causing silent failures that degrade performance without obvious errors:

**Why Consistency is Critical:**

1. **Feature Space Mismatch:**

Training time:
\`\`\`python
# Preprocessing accidentally different
text = "The U.S. economy grew 3% last quarter."
tokens = text.lower().split()  # ['the', 'u.s.', 'economy', ...]
\`\`\`

Inference time:
\`\`\`python
# Different preprocessing (forgot lowercase)
text = "The U.S. economy grew 3% last quarter."
tokens = text.split()  # ['The', 'U.S.', 'economy', ...]  # Different!
\`\`\`

Result: "The" and "U.S." are OOV (out-of-vocabulary) because training only saw lowercase versions. Model receives unexpected inputs.

2. **Vocabulary Drift:**

Training vocabulary built with stopword removal:
- Vocabulary: {economy, grew, quarter, ...}

Inference without stopword removal:
- Input: {the, economy, grew, quarter, ...}
- "the" is OOV → replaced with <UNK> → noise

3. **Statistical Distribution Shift:**

If training removes 40% of tokens (stopwords) but inference removes 20%, the model receives inputs with different statistical properties than it learned from.

4. **Cascade Effects:**

Small preprocessing inconsistencies → OOV tokens → incorrect predictions → cumulative error over time → degraded system performance.

**Common Sources of Inconsistency:**

1. **Hardcoded preprocessing**: Different code for training and inference
2. **Library version changes**: NLTK update changes tokenization slightly
3. **Missing dependencies**: Inference environment missing spaCy model
4. **Configuration drift**: Settings changed during debugging, not reverted
5. **Manual preprocessing**: Data scientist manually cleaned training data, but automation does something different

**Production-Grade Solution:**

**1. Preprocessing as a Versioned Artifact:**

\`\`\`python
import pickle
import hashlib
import json
from typing import Dict, Any

class VersionedPreprocessor:
    """Preprocessing pipeline with versioning and serialization"""
    
    VERSION = "1.2.0"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.config_hash = self._compute_config_hash()
        
        # Initialize components based on config
        self.lowercase = config.get('lowercase', True)
        self.remove_stopwords = config.get('remove_stopwords', False)
        # ... other settings
        
        # Load models/resources
        if config.get('use_spacy', False):
            import spacy
            self.nlp = spacy.load(config['spacy_model'])
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for versioning"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def preprocess(self, text: str) -> str:
        """Apply preprocessing pipeline"""
        # Implementation...
        return processed_text
    
    def save(self, path: str):
        """Save preprocessor with metadata"""
        metadata = {
            'version': self.VERSION,
            'config': self.config,
            'config_hash': self.config_hash,
            'dependencies': {
                'spacy': spacy.__version__ if 'spacy' in self.config else None,
                'nltk': nltk.__version__ if 'nltk' in self.config else None,
            }
        }
        
        with open(f"{path}/preprocessor_v{self.VERSION}.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load(path: str):
        """Load preprocessor with version verification"""
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Loading preprocessor version {metadata['version']}")
        print(f"Config hash: {metadata['config_hash']}")
        
        with open(f"{path}/preprocessor_v{metadata['version']}.pkl", 'rb') as f:
            return pickle.load(f)

# Usage at training time
config = {
    'lowercase': True,
    'remove_stopwords': True,
    'lemmatize': True,
    'spacy_model': 'en_core_web_sm'
}

preprocessor = VersionedPreprocessor(config)
preprocessor.save('models/preprocessor_v1')

# Usage at inference time (months later)
loaded_preprocessor = VersionedPreprocessor.load('models/preprocessor_v1')
# Guaranteed to use EXACT same preprocessing
\`\`\`

**2. Configuration Management:**

\`\`\`yaml
# preprocessing_config.yaml
version: "1.2.0"

preprocessing:
  lowercase: true
  remove_html: true
  remove_urls: true
  remove_stopwords: false  # Changed from true - will trigger version bump
  lemmatization: true
  min_token_length: 2

dependencies:
  spacy_model: "en_core_web_sm==3.5.0"  # Pin exact version
  nltk_data:
    - "punkt"
    - "wordnet"

# Track why changes were made
changelog:
  - version: "1.2.0"
    date: "2024-01-15"
    changes: "Disabled stopword removal for better sentiment detection"
    performance_impact: "+3.2% F1 on test set"
\`\`\`

**3. Testing Strategy:**

\`\`\`python
import unittest

class TestPreprocessorConsistency(unittest.TestCase):
    """Test preprocessing consistency"""
    
    def setUp(self):
        # Load training preprocessor
        self.train_preprocessor = VersionedPreprocessor.load('training_artifacts/')
        
        # Load inference preprocessor
        self.inference_preprocessor = VersionedPreprocessor.load('production/')
    
    def test_version_match(self):
        """Ensure versions match"""
        self.assertEqual(
            self.train_preprocessor.VERSION,
            self.inference_preprocessor.VERSION
        )
    
    def test_config_hash_match(self):
        """Ensure configurations are identical"""
        self.assertEqual(
            self.train_preprocessor.config_hash,
            self.inference_preprocessor.config_hash
        )
    
    def test_preprocessing_deterministic(self):
        """Ensure same input always produces same output"""
        test_texts = [
            "The quick brown fox",
            "Hello, World! 123",
            "machine learning and AI"
        ]
        
        for text in test_texts:
            result1 = self.train_preprocessor.preprocess(text)
            result2 = self.train_preprocessor.preprocess(text)
            self.assertEqual(result1, result2)
    
    def test_train_inference_match(self):
        """Ensure training and inference preprocessing match"""
        test_cases = [
            "The U.S. economy grew 3% last quarter.",
            "Apple Inc. released new iPhone models.",
            "I won't be attending the meeting.",
        ]
        
        for text in test_cases:
            train_result = self.train_preprocessor.preprocess(text)
            inference_result = self.inference_preprocessor.preprocess(text)
            
            self.assertEqual(
                train_result, 
                inference_result,
                f"Mismatch for: {text}"
            )
    
    def test_edge_cases(self):
        """Test edge cases that might break"""
        edge_cases = [
            "",  # Empty string
            "   ",  # Only whitespace
            "!!!",  # Only punctuation
            "😀🚀",  # Only emojis
            "a" * 10000,  # Very long
        ]
        
        for text in edge_cases:
            try:
                result = self.inference_preprocessor.preprocess(text)
                # Should not crash
                self.assertIsInstance(result, str)
            except Exception as e:
                self.fail(f"Failed on edge case '{text[:50]}': {e}")

# Run tests in CI/CD pipeline
if __name__ == '__main__':
    unittest.main()
\`\`\`

**4. Monitoring in Production:**

\`\`\`python
import logging
from collections import Counter

class MonitoredPreprocessor:
    def __init__(self, base_preprocessor):
        self.preprocessor = base_preprocessor
        self.oov_counter = Counter()
        self.input_stats = []
    
    def preprocess(self, text: str) -> str:
        # Track input characteristics
        self.input_stats.append({
            'length': len(text),
            'tokens': len(text.split()),
        })
        
        # Preprocess
        result = self.preprocessor.preprocess(text)
        
        # Track OOV if using vocabulary
        if hasattr(self, 'vocabulary'):
            tokens = result.split()
            for token in tokens:
                if token not in self.vocabulary:
                    self.oov_counter[token] += 1
        
        # Alert if too many OOV
        if len(self.oov_counter) > 100:
            logging.warning(f"High OOV rate: {len(self.oov_counter)} unique OOV tokens")
            logging.warning(f"Top OOV: {self.oov_counter.most_common(10)}")
        
        return result
\`\`\`

**5. CI/CD Integration:**

\`\`\`yaml
# .github/workflows/test_preprocessing.yml
name: Test Preprocessing Consistency

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_sm
      
      - name: Run preprocessing tests
        run: |
          python -m pytest tests/test_preprocessing.py -v
      
      - name: Verify preprocessor versions
        run: |
          python scripts/verify_preprocessor_consistency.py
      
      - name: Check for config drift
        run: |
          python scripts/check_config_drift.py
\`\`\`

**Best Practices Summary:**

1. **Version everything**: Preprocessor code, config, dependencies
2. **Serialize preprocessor**: Save fitted preprocessor object, not just code
3. **Pin dependencies**: Exact library versions in requirements.txt
4. **Comprehensive testing**: Unit tests for consistency
5. **Configuration as code**: YAML configs in version control
6. **Monitor in production**: Track OOV rates and input distributions
7. **Documentation**: Document why preprocessing choices were made
8. **Automate validation**: CI/CD checks for consistency

**Key Insight:**

Preprocessing is part of your model. Treat it with the same rigor as model architecture and hyperparameters. Version it, test it, and monitor it in production. Many production NLP failures are due to preprocessing inconsistencies, not model issues.`,
    keyPoints: [
      'Preprocessing inconsistency causes silent failures and performance degradation',
      'Small differences (lowercase, stopwords) create feature space mismatches',
      'Serialize and version preprocessor object, not just code',
      'Pin exact dependency versions to prevent library updates breaking preprocessing',
      'Comprehensive testing: version match, determinism, edge cases',
      'Monitor OOV rates in production to detect preprocessing drift',
      'Preprocessing is part of the model - treat it with same rigor',
    ],
  },
];
