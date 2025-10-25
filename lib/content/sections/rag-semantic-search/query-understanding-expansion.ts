export const queryUnderstandingExpansion = {
  title: 'Query Understanding & Expansion',
  content: `
# Query Understanding & Expansion

## Introduction

User queries are often incomplete, ambiguous, or poorly phrased. Query understanding and expansion techniques transform imperfect queries into effective retrieval operations, significantly improving RAG system performance.

In this comprehensive section, we'll explore NLP techniques, query rewriting, expansion strategies, and intent classification to maximize retrieval quality.

## Why Query Processing Matters

Raw user queries have limitations:

\`\`\`python
# Problems with raw queries:
query1 = "ml algos"  # Abbreviated, informal
query2 = "How do I train models?"  # Vague, missing context
query3 = "python async await syntax error help"  # Keywords only, no structure

# After processing:
# query1 → "machine learning algorithms"
# query2 → "How to train machine learning models in Python?"
# query3 → "Debugging async/await syntax errors in Python"
\`\`\`

### Benefits of Query Processing

1. **Better Retrieval**: More relevant documents retrieved
2. **Intent Understanding**: Know what user actually wants
3. **Synonym Expansion**: Match different phrasings
4. **Disambiguation**: Handle multiple meanings
5. **Recall Improvement**: Don't miss relevant docs due to phrasing

## Query Cleaning and Normalization

Basic preprocessing improves consistency:

\`\`\`python
import re
from typing import List

class QueryPreprocessor:
    """
    Clean and normalize queries for better retrieval.
    """
    
    def __init__(self):
        self.common_expansions = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "dl": "deep learning",
            "nlp": "natural language processing",
            "cv": "computer vision",
            "nn": "neural network",
            "llm": "large language model",
        }
    
    def clean (self, query: str) -> str:
        """
        Basic query cleaning.
        
        Args:
            query: Raw user query
        
        Returns:
            Cleaned query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub (r'\s+', ' ', query).strip()
        
        # Remove special characters (keep alphanumeric and spaces)
        query = re.sub (r'[^a-z0-9\s]', ', query)
        
        return query
    
    def expand_abbreviations (self, query: str) -> str:
        """
        Expand common abbreviations.
        
        Args:
            query: Query with potential abbreviations
        
        Returns:
            Query with expanded terms
        """
        words = query.split()
        expanded_words = []
        
        for word in words:
            if word in self.common_expansions:
                expanded_words.append (self.common_expansions[word])
            else:
                expanded_words.append (word)
        
        return ' '.join (expanded_words)
    
    def preprocess (self, query: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            query: Raw query
        
        Returns:
            Preprocessed query
        """
        query = self.clean (query)
        query = self.expand_abbreviations (query)
        return query


# Example usage
preprocessor = QueryPreprocessor()

raw_queries = [
    "ML algos!!!",
    "  python   async  patterns  ",
    "What\'s the best NLP model?"
]

for raw in raw_queries:
    processed = preprocessor.preprocess (raw)
    print(f"Raw: '{raw}'")
    print(f"Processed: '{processed}'\\n")
\`\`\`

## Named Entity Recognition (NER)

Extract entities to understand query intent:

\`\`\`python
import spacy
from typing import List, Dict

class QueryEntityExtractor:
    """
    Extract named entities from queries.
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize with spaCy model.
        
        Args:
            model: spaCy model name
        """
        try:
            self.nlp = spacy.load (model)
        except OSError:
            # Model not found, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load (model)
    
    def extract_entities (self, query: str) -> List[Dict]:
        """
        Extract named entities from query.
        
        Args:
            query: User query
        
        Returns:
            List of entities with types
        """
        doc = self.nlp (query)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def extract_keywords (self, query: str) -> List[str]:
        """
        Extract important keywords (nouns, proper nouns).
        
        Args:
            query: User query
        
        Returns:
            List of keywords
        """
        doc = self.nlp (query)
        
        keywords = []
        for token in doc:
            # Extract nouns and proper nouns
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                keywords.append (token.text.lower())
        
        return keywords


# Example usage
extractor = QueryEntityExtractor()

query = "How to use TensorFlow for image classification with ResNet?"
entities = extractor.extract_entities (query)
keywords = extractor.extract_keywords (query)

print(f"Query: {query}")
print(f"\\nEntities:")
for ent in entities:
    print(f"  {ent['text']} ({ent['label']})")

print(f"\\nKeywords: {keywords}")
\`\`\`

## Intent Classification

Understand what type of information the user seeks:

\`\`\`python
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class QueryIntentClassifier:
    """
    Classify query intent to route appropriately.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer (max_features=1000)
        self.classifier = MultinomialNB()
        self.intent_labels = []
        self.is_trained = False
    
    def train(
        self,
        queries: List[str],
        intents: List[str]
    ):
        """
        Train intent classifier.
        
        Args:
            queries: Training queries
            intents: Intent labels
        """
        # Vectorize queries
        X = self.vectorizer.fit_transform (queries)
        
        # Store unique intents
        self.intent_labels = list (set (intents))
        
        # Train classifier
        self.classifier.fit(X, intents)
        self.is_trained = True
    
    def predict_intent (self, query: str) -> Dict:
        """
        Predict intent for a query.
        
        Args:
            query: User query
        
        Returns:
            Intent with confidence score
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")
        
        # Vectorize query
        X = self.vectorizer.transform([query])
        
        # Predict
        intent = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Get confidence for predicted intent
        intent_idx = self.intent_labels.index (intent)
        confidence = probabilities[intent_idx]
        
        return {
            "intent": intent,
            "confidence": float (confidence),
            "all_probabilities": {
                label: float (prob)
                for label, prob in zip (self.intent_labels, probabilities)
            }
        }


# Example: Train on sample data
classifier = QueryIntentClassifier()

# Training data
training_queries = [
    "How do I train a model?",
    "What is machine learning?",
    "Show me examples of neural networks",
    "Explain gradient descent",
    "Tutorial for PyTorch",
    "What\'s the definition of overfitting?",
]

training_intents = [
    "how-to",
    "definition",
    "examples",
    "explanation",
    "tutorial",
    "definition",
]

classifier.train (training_queries, training_intents)

# Predict intent
test_query = "How can I implement a CNN in TensorFlow?"
result = classifier.predict_intent (test_query)

print(f"Query: {test_query}")
print(f"Intent: {result['intent']} ({result['confidence']:.2f})")
\`\`\`

## Query Expansion with Synonyms

Expand queries with synonyms to improve recall:

\`\`\`python
from nltk.corpus import wordnet
import nltk

# Download WordNet data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class SynonymExpander:
    """
    Expand queries using synonyms.
    """
    
    def __init__(self, max_synonyms: int = 3):
        """
        Initialize synonym expander.
        
        Args:
            max_synonyms: Maximum synonyms per word
        """
        self.max_synonyms = max_synonyms
    
    def get_synonyms (self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Word to find synonyms for
        
        Returns:
            List of synonyms
        """
        synonyms = set()
        
        for syn in wordnet.synsets (word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.add (synonym.lower())
        
        return list (synonyms)[:self.max_synonyms]
    
    def expand_query (self, query: str) -> List[str]:
        """
        Generate query variations with synonyms.
        
        Args:
            query: Original query
        
        Returns:
            List of query variations
        """
        words = query.lower().split()
        variations = [query]  # Include original
        
        # Generate variations by replacing one word at a time
        for i, word in enumerate (words):
            synonyms = self.get_synonyms (word)
            
            for synonym in synonyms:
                # Create variation with this synonym
                new_words = words.copy()
                new_words[i] = synonym
                variation = ' '.join (new_words)
                variations.append (variation)
        
        return variations


# Example usage
expander = SynonymExpander (max_synonyms=2)

query = "quick way to learn python"
variations = expander.expand_query (query)

print(f"Original: {query}\\n")
print("Variations:")
for var in variations[:5]:  # Show first 5
    print(f"  - {var}")
\`\`\`

## Query Rewriting with LLMs

Use LLMs to rewrite queries for better retrieval:

\`\`\`python
from openai import OpenAI

client = OpenAI()

class LLMQueryRewriter:
    """
    Rewrite queries using LLM for improved retrieval.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI()
    
    def rewrite (self, query: str) -> str:
        """
        Rewrite query to be more specific and searchable.
        
        Args:
            query: Original query
        
        Returns:
            Rewritten query
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite the user's query to be more specific, complete, and optimized for document search. Keep it concise (1-2 sentences)."
                },
                {
                    "role": "user",
                    "content": f"Query: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_variations(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
        
        Returns:
            List of query variations
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"Generate {num_variations} different ways to search for the following query. Each should capture the same intent but with different wording. Return only the queries, one per line."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        content = response.choices[0].message.content
        variations = [v.strip() for v in content.split('\\n') if v.strip()]
        
        return variations[:num_variations]


# Example usage
rewriter = LLMQueryRewriter()

original = "ml model accuracy"
rewritten = rewriter.rewrite (original)
variations = rewriter.generate_variations (original, num_variations=3)

print(f"Original: {original}")
print(f"\\nRewritten: {rewritten}")
print(f"\\nVariations:")
for i, var in enumerate (variations, 1):
    print(f"  {i}. {var}")
\`\`\`

## Multi-Query Generation for Better Recall

Generate multiple query perspectives:

\`\`\`python
class MultiQueryGenerator:
    """
    Generate multiple query variations for improved recall.
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_multi_queries(
        self,
        query: str,
        num_queries: int = 5
    ) -> List[str]:
        """
        Generate multiple query variations.
        
        Args:
            query: Original query
            num_queries: Number of variations
        
        Returns:
            List of query variations
        """
        prompt = f"""Generate {num_queries} different search queries that capture the same information need as: "{query}"

Each query should:
1. Use different wording and structure
2. Emphasize different aspects of the question
3. Be specific enough for document search

Return only the queries, numbered 1-{num_queries}."""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a search query expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        
        # Parse numbered list
        queries = []
        for line in content.split('\\n'):
            line = line.strip()
            # Remove numbering (1., 2., etc.)
            if line and line[0].isdigit():
                query_text = re.sub (r'^\\d+\\.\\s*', ', line)
                queries.append (query_text)
        
        return queries[:num_queries]
    
    def generate_with_perspectives (self, query: str) -> Dict[str, str]:
        """
        Generate queries from different perspectives.
        
        Args:
            query: Original query
        
        Returns:
            Dict of perspective -> query
        """
        perspectives = {
            "beginner": "Explain in simple terms",
            "technical": "Technical deep-dive",
            "practical": "Practical how-to guide",
            "conceptual": "Theoretical understanding",
        }
        
        result = {}
        
        for perspective, instruction in perspectives.items():
            prompt = f"{instruction}: {query}"
            rewritten = self._rewrite_single (prompt)
            result[perspective] = rewritten
        
        return result
    
    def _rewrite_single (self, prompt: str) -> str:
        """Helper to rewrite single query."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()


# Example
generator = MultiQueryGenerator()

query = "machine learning model evaluation"
multi_queries = generator.generate_multi_queries (query, num_queries=3)
perspectives = generator.generate_with_perspectives (query)

print(f"Original: {query}\\n")
print("Multi-queries:")
for i, mq in enumerate (multi_queries, 1):
    print(f"  {i}. {mq}")

print(f"\\nDifferent perspectives:")
for persp, q in perspectives.items():
    print(f"  {persp}: {q}")
\`\`\`

## Complete Query Processing Pipeline

Production-ready query processing:

\`\`\`python
from typing import List, Dict, Optional

class QueryProcessor:
    """
    Complete query processing pipeline.
    """
    
    def __init__(
        self,
        use_ner: bool = True,
        use_intent: bool = True,
        use_expansion: bool = True,
        use_llm_rewrite: bool = False
    ):
        """
        Initialize query processor.
        
        Args:
            use_ner: Enable named entity recognition
            use_intent: Enable intent classification
            use_expansion: Enable query expansion
            use_llm_rewrite: Enable LLM-based rewriting
        """
        self.use_ner = use_ner
        self.use_intent = use_intent
        self.use_expansion = use_expansion
        self.use_llm_rewrite = use_llm_rewrite
        
        # Initialize components
        self.preprocessor = QueryPreprocessor()
        
        if use_ner:
            self.entity_extractor = QueryEntityExtractor()
        
        if use_expansion:
            self.synonym_expander = SynonymExpander()
        
        if use_llm_rewrite:
            self.llm_rewriter = LLMQueryRewriter()
    
    def process (self, query: str) -> Dict:
        """
        Process query through complete pipeline.
        
        Args:
            query: Raw user query
        
        Returns:
            Processed query with metadata
        """
        result = {
            "original": query,
            "cleaned": None,
            "entities": [],
            "intent": None,
            "variations": [],
            "rewritten": None,
        }
        
        # Step 1: Clean and normalize
        cleaned = self.preprocessor.preprocess (query)
        result["cleaned"] = cleaned
        
        # Step 2: Extract entities
        if self.use_ner:
            result["entities"] = self.entity_extractor.extract_entities (query)
        
        # Step 3: Classify intent
        if self.use_intent and hasattr (self, 'intent_classifier'):
            result["intent"] = self.intent_classifier.predict_intent (cleaned)
        
        # Step 4: Generate variations
        if self.use_expansion:
            result["variations"] = self.synonym_expander.expand_query (cleaned)
        
        # Step 5: LLM rewrite (optional)
        if self.use_llm_rewrite:
            result["rewritten"] = self.llm_rewriter.rewrite (cleaned)
        
        return result
    
    def get_search_queries (self, processed: Dict) -> List[str]:
        """
        Get all query variations for search.
        
        Args:
            processed: Processed query dict
        
        Returns:
            List of queries to search with
        """
        queries = [processed["cleaned"]]
        
        if processed["rewritten"]:
            queries.append (processed["rewritten"])
        
        if processed["variations"]:
            queries.extend (processed["variations"][:3])  # Top 3 variations
        
        return queries


# Example usage
processor = QueryProcessor(
    use_ner=True,
    use_expansion=True,
    use_llm_rewrite=False  # Set to True if you have API key
)

query = "How do I train ML models?"
processed = processor.process (query)

print(f"Original: {processed['original']}")
print(f"Cleaned: {processed['cleaned']}")
print(f"\\nEntities: {processed['entities']}")
print(f"\\nQuery variations for search:")
search_queries = processor.get_search_queries (processed)
for sq in search_queries:
    print(f"  - {sq}")
\`\`\`

## Best Practices

### When to Use Each Technique

| Technique | Use When | Benefit |
|-----------|----------|---------|
| **Cleaning** | Always | Consistency |
| **Abbreviation Expansion** | Domain-specific queries | Better matching |
| **NER** | Entity-focused queries | Extract key terms |
| **Intent Classification** | Multiple search types | Better routing |
| **Synonym Expansion** | Improving recall | Find varied phrasings |
| **LLM Rewriting** | Complex/vague queries | Clarification |
| **Multi-Query** | Critical searches | Maximum recall |

### Performance Considerations

\`\`\`python
# Optimize query processing
class OptimizedQueryProcessor:
    """Query processor optimized for production."""
    
    def process (self, query: str) -> Dict:
        # 1. Fast path for simple queries
        if len (query.split()) <= 3 and query.isascii():
            return {"original": query, "cleaned": query.lower()}
        
        # 2. Cache processed queries
        cached = self.cache.get (query)
        if cached:
            return cached
        
        # 3. Full processing for complex queries
        result = self.full_process (query)
        self.cache.set (query, result)
        
        return result
\`\`\`

## Summary

Query processing dramatically improves RAG retrieval:

- **Cleaning**: Normalize queries for consistency
- **NER**: Extract important entities
- **Intent Classification**: Understand what user wants
- **Expansion**: Generate variations for better recall
- **LLM Rewriting**: Clarify vague queries
- **Multi-Query**: Multiple perspectives for comprehensive search

**Key Takeaway:** Invest in query processing early. It's often easier and cheaper than improving the retrieval model itself.

**Production Pattern:**
1. Always clean and normalize
2. Extract entities for filtering
3. Expand for important queries
4. Use LLM rewriting sparingly (cost)
5. Cache processed queries
`,
};
