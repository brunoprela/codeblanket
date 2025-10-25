/**
 * Section: Text Preprocessing
 * Module: Natural Language Processing
 *
 * Covers tokenization, text cleaning, normalization, stemming, lemmatization, and preprocessing pipelines
 */

export const textPreprocessingSection = {
  id: 'text-preprocessing',
  title: 'Text Preprocessing',
  content: `
# Text Preprocessing

## Introduction

Text preprocessing is the foundation of any Natural Language Processing (NLP) pipeline. Raw text data is messy—it contains inconsistent formatting, punctuation, capitalization, and other noise that can hinder machine learning models. Proper preprocessing transforms this raw text into a clean, standardized format that ML algorithms can effectively learn from.

**Why Text Preprocessing Matters:**
- **Reduces noise**: Removes irrelevant information that doesn't contribute to meaning
- **Standardizes format**: Ensures consistent input for models
- **Reduces vocabulary size**: Improves model efficiency and generalization
- **Improves performance**: Can dramatically boost model accuracy
- **Enables feature extraction**: Prepares text for vectorization

The challenge is finding the right balance—aggressive preprocessing can remove important information, while minimal preprocessing can leave too much noise.

## Tokenization

Tokenization is the process of breaking text into individual units (tokens), typically words or subwords. It\'s the first step in almost every NLP pipeline.

### Word Tokenization

\`\`\`python
import nltk
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, WordPunctTokenizer

# Download required NLTK data
nltk.download('punkt', quiet=True)

text = "Hello! This is an example sentence. Let\'s tokenize it."

# NLTK word tokenizer (handles punctuation well)
tokens_nltk = word_tokenize (text)
print(f"NLTK tokens: {tokens_nltk}")
# ['Hello', '!', 'This', 'is', 'an', 'example', 'sentence', '.', 'Let', "'s", 'tokenize', 'it', '.']

# Whitespace tokenizer (simple split on whitespace)
whitespace_tokenizer = WhitespaceTokenizer()
tokens_ws = whitespace_tokenizer.tokenize (text)
print(f"\\nWhitespace tokens: {tokens_ws}")
# ['Hello!', 'This', 'is', 'an', 'example', 'sentence.', "Let's", 'tokenize', 'it.']

# Word-punct tokenizer (splits on punctuation)
wordpunct_tokenizer = WordPunctTokenizer()
tokens_wp = wordpunct_tokenizer.tokenize (text)
print(f"\\nWord-punct tokens: {tokens_wp}")
# ['Hello', '!', 'This', 'is', 'an', 'example', 'sentence', '.', 'Let', "'", 's', 'tokenize', 'it', '.']
\`\`\`

### Sentence Tokenization

\`\`\`python
from nltk.tokenize import sent_tokenize

text = """
Natural language processing is fascinating. It enables computers to understand text! 
Dr. Smith published a paper on NLP in 2023. The results were impressive.
"""

sentences = sent_tokenize (text)
print(f"Number of sentences: {len (sentences)}")
for i, sent in enumerate (sentences, 1):
    print(f"{i}. {sent.strip()}")
\`\`\`

**Output:**
\`\`\`
Number of sentences: 3
1. Natural language processing is fascinating.
2. It enables computers to understand text!
3. Dr. Smith published a paper on NLP in 2023.
4. The results were impressive.
\`\`\`

### Advanced Tokenization with spaCy

spaCy provides industrial-strength tokenization with linguistic awareness:

\`\`\`python
import spacy

# Load English model
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is looking at buying U.K. startup for $1 billion. Isn't that amazing?"
doc = nlp (text)

# spaCy tokenization (handles contractions, entities, and abbreviations well)
tokens = [token.text for token in doc]
print(f"spaCy tokens: {tokens}")
# ['Apple', 'Inc.', 'is', 'looking', 'at', 'buying', 'U.K.', 'startup', 'for', '$', '1', 'billion', '.', 'Is', "n't", 'that', 'amazing', '?']

# Token properties
print("\\nToken details:")
for token in doc[:5]:
    print(f"Text: {token.text:10} | Lemma: {token.lemma_:10} | POS: {token.pos_}")
\`\`\`

## Text Cleaning

### Lowercasing

\`\`\`python
text = "The Quick BROWN Fox Jumps Over THE Lazy Dog"
lowercased = text.lower()
print(f"Original: {text}")
print(f"Lowercased: {lowercased}")

# When NOT to lowercase
# - Named entities: "Apple" (company) vs "apple" (fruit)
# - Acronyms: "US" vs "us"
# - Case-sensitive tasks: sentiment ("GREAT!" has more emphasis)
\`\`\`

### Removing Punctuation

\`\`\`python
import string
import re

text = "Hello, world! How are you? I'm fine, thanks."

# Method 1: Using string.punctuation
no_punct = text.translate (str.maketrans(', ', string.punctuation))
print(f"No punctuation: {no_punct}")
# "Hello world How are you Im fine thanks"

# Method 2: Using regex (keeps contractions)
no_punct_regex = re.sub (r'[^\\w\\s']', ', text)
print(f"Regex (keeps apostrophes): {no_punct_regex}")
# "Hello world How are you I'm fine thanks"

# Method 3: Selective removal
def remove_punct_except (text, keep='.,!?'):
    punct_to_remove = '.join([c for c in string.punctuation if c not in keep])
    return text.translate (str.maketrans(', ', punct_to_remove))

selective = remove_punct_except (text)
print(f"Selective removal: {selective}")
\`\`\`

### Removing Numbers

\`\`\`python
text = "I have 3 cats and 2 dogs. They cost $150 total."

# Remove all numbers
no_numbers = re.sub (r'\\d+', ', text)
print(f"No numbers: {no_numbers}")
# "I have  cats and  dogs. They cost $ total."

# Remove numbers but keep currency
no_numbers_keep_currency = re.sub (r'(?<!\\$)\\b\\d+\\b', ', text)
print(f"Keep currency: {no_numbers_keep_currency}")

# Replace numbers with <NUM> token
numbers_replaced = re.sub (r'\\d+', '<NUM>', text)
print(f"Numbers replaced: {numbers_replaced}")
# "I have <NUM> cats and <NUM> dogs. They cost $<NUM> total."
\`\`\`

### Removing Whitespace

\`\`\`python
text = "  This   has    irregular    spacing.  \\n\\n  "

# Strip leading/trailing whitespace
stripped = text.strip()
print(f"Stripped: '{stripped}'")

# Normalize internal whitespace
normalized = ' '.join (text.split())
print(f"Normalized: '{normalized}'")
# "This has irregular spacing."
\`\`\`

### Removing HTML and URLs

\`\`\`python
import re
from html import unescape

text = """
<p>Check out this <a href="https://example.com">link</a>!</p>
Email me at user@example.com or visit http://website.com
"""

# Remove HTML tags
no_html = re.sub (r'<.*?>', ', text)
print(f"No HTML: {no_html}")

# Unescape HTML entities
html_text = "AT&amp;T and &quot;quotes&quot; &lt;tags&gt;"
unescaped = unescape (html_text)
print(f"\\nUnescaped: {unescaped}")
# "AT&T and "quotes" <tags>"

# Remove URLs
no_urls = re.sub (r'http\\S+|www\\S+', ', text)
print(f"\\nNo URLs: {no_urls}")

# Remove email addresses
no_emails = re.sub (r'\\S+@\\S+', ', text)
print(f"No emails: {no_emails}")
\`\`\`

## Stop Words Removal

Stop words are common words (like "the", "is", "at") that appear frequently but often don't contribute much to meaning.

\`\`\`python
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

text = "This is a sample sentence demonstrating the removal of stop words"
tokens = text.lower().split()

# NLTK stopwords
stop_words = set (stopwords.words('english'))
print(f"Number of stopwords: {len (stop_words)}")
print(f"Sample stopwords: {list (stop_words)[:10]}")

# Remove stopwords
filtered_tokens = [word for word in tokens if word not in stop_words]
print(f"\\nOriginal: {tokens}")
print(f"Filtered: {filtered_tokens}")
# ['sample', 'sentence', 'demonstrating', 'removal', 'stop', 'words']

# Custom stopwords
custom_stopwords = stop_words.union({'sample', 'demonstrating'})
custom_filtered = [word for word in tokens if word not in custom_stopwords]
print(f"Custom filtered: {custom_filtered}")
# ['sentence', 'removal', 'stop', 'words']
\`\`\`

**When NOT to Remove Stopwords:**
- **Sentiment analysis**: "not good" vs "good" (completely different meanings)
- **Question answering**: "Who is", "What is" are important for context
- **Machine translation**: Stopwords are essential for grammatical structure
- **Modern transformers**: BERT and GPT benefit from all context, including stopwords

## Stemming

Stemming reduces words to their root form using heuristic rules. It\'s fast but can be imprecise.

\`\`\`python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

words = ["running", "runs", "ran", "runner", "easily", "fairly", "studying", "studies"]

# Porter Stemmer (most common)
porter = PorterStemmer()
porter_stems = [porter.stem (word) for word in words]
print(f"Porter: {porter_stems}")
# ['run', 'run', 'ran', 'runner', 'easili', 'fairli', 'studi', 'studi']

# Lancaster Stemmer (more aggressive)
lancaster = LancasterStemmer()
lancaster_stems = [lancaster.stem (word) for word in words]
print(f"Lancaster: {lancaster_stems}")
# ['run', 'run', 'ran', 'run', 'easy', 'fair', 'study', 'study']

# Snowball Stemmer (supports multiple languages)
snowball = SnowballStemmer('english')
snowball_stems = [snowball.stem (word) for word in words]
print(f"Snowball: {snowball_stems}")
# ['run', 'run', 'ran', 'runner', 'easili', 'fairli', 'studi', 'studi']

# Comparison
print("\\nWord      | Porter | Lancaster | Snowball")
print("-" * 45)
for word in words:
    print(f"{word:10} | {porter.stem (word):6} | {lancaster.stem (word):9} | {snowball.stem (word):8}")
\`\`\`

### Stemming Issues

\`\`\`python
# Over-stemming (conflating unrelated words)
words_overstem = ["university", "universe", "universal"]
for word in words_overstem:
    print(f"{word} -> {porter.stem (word)}")
# All become "univers" even though they have different meanings

# Under-stemming (not conflating related words)
words_understem = ["data", "datum", "database"]
for word in words_understem:
    print(f"{word} -> {porter.stem (word)}")
# "data" -> "data", "datum" -> "datum", "database" -> "databas"
\`\`\`

## Lemmatization

Lemmatization reduces words to their dictionary form (lemma) using morphological analysis. It\'s slower but more accurate than stemming.

\`\`\`python
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "runner", "easily", "fairly", "better", "worse", "geese", "mice"]

# Lemmatization (default assumes noun)
lemmas = [lemmatizer.lemmatize (word) for word in words]
print(f"Default (noun): {lemmas}")
# ['running', 'run', 'ran', 'runner', 'easily', 'fairly', 'better', 'worse', 'goose', 'mouse']

# Lemmatization with POS tags (more accurate)
from nltk import pos_tag

# POS-aware lemmatization
def get_wordnet_pos (treebank_tag):
    """Convert treebank POS tag to WordNet POS tag"""
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun

nltk.download('averaged_perceptron_tagger', quiet=True)

sentence = "The striped bats are hanging on their feet for best"
tokens = word_tokenize (sentence)
pos_tags = pos_tag (tokens)

print("\\nPOS-aware lemmatization:")
for token, pos in pos_tags:
    wordnet_pos = get_wordnet_pos (pos)
    lemma = lemmatizer.lemmatize (token.lower(), pos=wordnet_pos)
    print(f"{token:10} ({pos:4}) -> {lemma}")
\`\`\`

### Stemming vs Lemmatization Comparison

\`\`\`python
import time

words_test = ["studies", "studying", "studied", "studies", "better", "good", "running", "ran"]

print("Word      | Stemming | Lemmatization")
print("-" * 40)
for word in words_test:
    stem = porter.stem (word)
    lemma = lemmatizer.lemmatize (word, pos='v')
    print(f"{word:10} | {stem:8} | {lemma:13}")

# Performance comparison
test_text = " ".join (words_test * 1000)
tokens = test_text.split()

# Stemming speed
start = time.time()
stems = [porter.stem (word) for word in tokens]
stem_time = time.time() - start

# Lemmatization speed
start = time.time()
lemmas = [lemmatizer.lemmatize (word) for word in tokens]
lemma_time = time.time() - start

print(f"\\nStemming time: {stem_time:.4f}s")
print(f"Lemmatization time: {lemma_time:.4f}s")
print(f"Speedup: {lemma_time/stem_time:.2f}x faster for stemming")
\`\`\`

## Text Normalization

\`\`\`python
# Handling contractions
contractions_dict = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am"
}

def expand_contractions (text):
    for contraction, expansion in contractions_dict.items():
        text = text.replace (contraction, expansion)
    return text

text = "I won't be there. She can't go. They're happy and we'll join."
expanded = expand_contractions (text)
print(f"Original: {text}")
print(f"Expanded: {expanded}")

# Handling spelling variations
import re

def normalize_british_to_american (text):
    """Convert British to American spelling"""
    replacements = {
        r'colour': 'color',
        r'flavour': 'flavor',
        r'analyse': 'analyze',
        r'centre': 'center',
        r'behaviour': 'behavior'
    }
    for british, american in replacements.items():
        text = re.sub (british, american, text, flags=re.IGNORECASE)
    return text

text_british = "The colour of the centre was analysed."
text_american = normalize_british_to_american (text_british)
print(f"\\nBritish: {text_british}")
print(f"American: {text_american}")

# Handling repetitions (social media)
def normalize_repetitions (text):
    """Reduce character repetitions to max 2"""
    return re.sub (r'(\\w)\\1{2,}', r'\\1\\1', text)

text_social = "Thaaaaaaat was sooooo goooood!!! 😂😂😂"
normalized = normalize_repetitions (text_social)
print(f"\\nSocial media text: {text_social}")
print(f"Normalized: {normalized}")
\`\`\`

## Complete Preprocessing Pipeline

\`\`\`python
import re
import string
from typing import List, Optional
import spacy

class TextPreprocessor:
    """Comprehensive text preprocessing pipeline"""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        stemming: bool = False,
        lemmatization: bool = True,
        min_token_length: int = 2
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.min_token_length = min_token_length
        
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        
        # Initialize stemmer if needed
        if self.stemming:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        
        # Load stopwords if needed
        if self.remove_stopwords:
            from nltk.corpus import stopwords
            self.stop_words = set (stopwords.words('english'))
    
    def clean_text (self, text: str) -> str:
        """Basic text cleaning"""
        # Remove HTML tags
        if self.remove_html:
            text = re.sub (r'<.*?>', ', text)
            from html import unescape
            text = unescape (text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub (r'http\\S+|www\\S+|https\\S+', ', text)
            text = re.sub (r'\\S+@\\S+', ', text)  # Remove emails too
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate (str.maketrans(', ', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub (r'\\d+', ', text)
        
        # Normalize whitespace
        text = ' '.join (text.split())
        
        return text
    
    def tokenize_and_process (self, text: str) -> List[str]:
        """Tokenize and apply word-level processing"""
        doc = self.nlp (text)
        tokens = []
        
        for token in doc:
            # Skip short tokens
            if len (token.text) < self.min_token_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token.text.lower() in self.stop_words:
                continue
            
            # Get processed token
            if self.lemmatization:
                processed_token = token.lemma_
            elif self.stemming:
                processed_token = self.stemmer.stem (token.text)
            else:
                processed_token = token.text
            
            tokens.append (processed_token)
        
        return tokens
    
    def preprocess (self, text: str, return_string: bool = False) -> Optional[List[str]]:
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text (text)
        
        # Tokenize and process
        tokens = self.tokenize_and_process (text)
        
        # Return as string or list
        if return_string:
            return ' '.join (tokens)
        return tokens

# Example usage
text = """
<p>Check out this article at https://example.com! 
The researchers found that 95% of the data was valuable. 
"This is amazing," said Dr. Smith. #NLP #MachineLearning</p>
"""

# Different preprocessing strategies
print("1. Minimal preprocessing (for transformers):")
preprocessor_minimal = TextPreprocessor(
    lowercase=True,
    remove_html=True,
    remove_urls=True,
    remove_punctuation=False,
    remove_stopwords=False,
    lemmatization=False
)
print(preprocessor_minimal.preprocess (text, return_string=True))

print("\\n2. Aggressive preprocessing (for classical ML):")
preprocessor_aggressive = TextPreprocessor(
    lowercase=True,
    remove_html=True,
    remove_urls=True,
    remove_punctuation=True,
    remove_numbers=True,
    remove_stopwords=True,
    lemmatization=True,
    min_token_length=3
)
print(preprocessor_aggressive.preprocess (text, return_string=True))

print("\\n3. Stemming approach:")
preprocessor_stem = TextPreprocessor(
    lowercase=True,
    remove_html=True,
    remove_urls=True,
    remove_punctuation=True,
    remove_stopwords=True,
    stemming=True,
    lemmatization=False
)
print(preprocessor_stem.preprocess (text, return_string=True))
\`\`\`

## Preprocessing for Different Tasks

\`\`\`python
# 1. Sentiment Analysis
# Keep: punctuation (!!!, ???), capitalization, emoticons
# Remove: HTML, URLs
sentiment_preprocessor = TextPreprocessor(
    lowercase=False,  # Keep case for emphasis
    remove_punctuation=False,  # Keep for emotion
    remove_stopwords=False,  # "not good" vs "good"
    lemmatization=False  # Keep tense ("loved" vs "will love")
)

sentiment_text = "I LOVED this product!!! Best purchase ever!"
print(f"Sentiment preprocessing: {sentiment_preprocessor.preprocess (sentiment_text, return_string=True)}")

# 2. Topic Modeling
# Remove: stopwords, punctuation
# Apply: lemmatization
topic_preprocessor = TextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    lemmatization=True,
    min_token_length=3
)

topic_text = "The researchers conducted extensive studies on neural networks and machine learning algorithms."
print(f"\\nTopic preprocessing: {topic_preprocessor.preprocess (topic_text, return_string=True)}")

# 3. Named Entity Recognition
# Keep: capitalization, punctuation
# Minimal preprocessing
ner_preprocessor = TextPreprocessor(
    lowercase=False,  # "Apple Inc." needs capitalization
    remove_punctuation=False,  # "Dr." needs punctuation
    remove_stopwords=False,
    lemmatization=False
)

ner_text = "Apple Inc. CEO Tim Cook announced the new iPhone in California."
print(f"\\nNER preprocessing: {ner_preprocessor.preprocess (ner_text, return_string=True)}")
\`\`\`

## Best Practices and Common Pitfalls

\`\`\`python
# Pitfall 1: Over-preprocessing
text = "not good at all"

over_preprocessed = TextPreprocessor(
    remove_stopwords=True,  # Removes "not"
    lemmatization=True
).preprocess (text, return_string=True)

print(f"Over-preprocessed: '{over_preprocessed}'")  # "good" - Wrong sentiment!

# Solution: Task-specific preprocessing
proper = TextPreprocessor(
    remove_stopwords=False,
    lemmatization=True
).preprocess (text, return_string=True)
print(f"Properly preprocessed: '{proper}'")

# Pitfall 2: Inconsistent preprocessing
# Training text
train_text = "The quick brown fox"
train_preprocessor = TextPreprocessor (lowercase=True, lemmatization=True)
train_processed = train_preprocessor.preprocess (train_text, return_string=True)

# Test text with different preprocessing (ERROR!)
test_text = "THE QUICK BROWN FOX"
test_processed = test_text.split()  # Forgot to preprocess!

print(f"\\nTrain: {train_processed}")
print(f"Test (wrong): {test_processed}")  # Different format!

# Solution: Save and reuse preprocessing pipeline
import pickle

# Save preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump (train_preprocessor, f)

# Load and reuse
with open('preprocessor.pkl', 'rb') as f:
    loaded_preprocessor = pickle.load (f)

test_processed_correct = loaded_preprocessor.preprocess (test_text, return_string=True)
print(f"Test (correct): {test_processed_correct}")
\`\`\`

## Summary

Text preprocessing is a critical but nuanced step in NLP:

### Key Takeaways:
1. **No universal recipe**: Different tasks require different preprocessing strategies
2. **Modern transformers**: Benefit from minimal preprocessing (keep punctuation, stopwords)
3. **Classical ML**: Benefits from aggressive preprocessing (remove noise, reduce vocabulary)
4. **Consistency is crucial**: Use identical preprocessing for training and inference
5. **Document decisions**: Track what preprocessing was applied for reproducibility

### Decision Matrix:

| Task | Lowercase | Remove Punct | Remove Stopwords | Lemmatization |
|------|-----------|--------------|------------------|---------------|
| Sentiment Analysis | ✗ | ✗ | ✗ | ✗ |
| Topic Modeling | ✓ | ✓ | ✓ | ✓ |
| Text Classification | ✓ | ✓ | ? | ✓ |
| NER | ✗ | ✗ | ✗ | ✗ |
| Machine Translation | ✗ | ✗ | ✗ | ✗ |
| Transformers (BERT, GPT) | ✗/✓ | ✗ | ✗ | ✗ |

### Modern Best Practices:
- **Experiment**: Try different preprocessing strategies and measure impact
- **Start minimal**: Add preprocessing only if it improves performance
- **Use version control**: Track preprocessing code alongside models
- **Consider subword tokenization**: BPE, WordPiece handle unknown words better
- **For transformers**: Let the model learn from rich context (minimal preprocessing)
`,
};
