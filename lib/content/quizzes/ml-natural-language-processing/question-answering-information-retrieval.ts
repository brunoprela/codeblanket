import { QuizQuestion } from '../../../types';

export const questionAnsweringQuiz: QuizQuestion[] = [
  {
    id: 'qa-ir-dq-1',
    question:
      'Explain the difference between extractive and generative question answering. What are the trade-offs between these approaches?',
    sampleAnswer: `Extractive and generative QA represent fundamentally different approaches to answering questions:

**Extractive QA:**

Finds answer spans within provided context:
- Input: Question + Context
- Output: Start and end positions in context
- Example: "When was the Eiffel Tower built?" → "from 1887 to 1889" (extracted from text)

**How it works:**1. Model reads question and context
2. Predicts start token position
3. Predicts end token position
4. Extracts span between positions

**Advantages:**
- Factually grounded (answer must be in text)
- No hallucination risk
- Interpretable (can show source)
- Faster inference (no generation needed)
- Works with smaller models

**Disadvantages:**
- Limited to information in context
- Cannot synthesize information
- Cannot rephrase answers
- Requires good context

**Generative QA:**

Generates free-form answers:
- Input: Question (+ optional context)
- Output: Generated text answer
- Example: "Why was the Eiffel Tower built?" → "The Eiffel Tower was built as the entrance arch for the 1889 World\'s Fair to celebrate the centennial of the French Revolution." (synthesized)

**How it works:**1. Encoder processes question and context
2. Decoder generates answer token-by-token
3. Can combine information from multiple sources
4. Can rephrase and summarize

**Advantages:**
- Can synthesize information
- Natural, fluent answers
- Can handle questions without explicit answers
- Can provide explanations

**Disadvantages:**
- Hallucination risk (may generate false information)
- Harder to verify factual accuracy
- Slower (generation is sequential)
- Requires larger models (GPT-3, T5)
- Less interpretable

**Trade-offs:**

**Use Extractive when:**
- Factual accuracy critical (legal, medical)
- Need source attribution
- Limited compute/latency constraints
- Context contains answer explicitly

**Use Generative when:**
- Need natural, fluent responses
- Synthesis required (multi-document)
- Questions need explanation
- Conversational AI applications

**Hybrid Approaches:**

Modern systems often combine both:
1. **Retrieval + Extractive**: Find documents, extract answer (BERT-based)
2. **Retrieval + Generative**: Find documents, generate answer (RAG - Retrieval Augmented Generation)
3. **Extractive → Generative**: Extract relevant spans, then generate fluent answer

**Example Comparison:**

Question: "What is machine learning?"

**Extractive (from context):**
- "a subset of artificial intelligence" (exact extraction)
- Pros: Factual, grounded
- Cons: May be incomplete or awkward phrasing

**Generative:**
- "Machine learning is a field of AI that enables computers to learn from data without being explicitly programmed" (synthesized)
- Pros: Complete, fluent explanation
- Cons: May add unsupported details

**Production Considerations:**

For mission-critical applications (medical diagnosis, legal advice):
- Use extractive QA
- Always show source context
- Verify factual accuracy

For user-facing applications (chatbots, assistants):
- Use generative or hybrid
- Implement hallucination detection
- Provide sources when possible

The trend is toward generative systems (ChatGPT, GPT-4) but extractive QA remains valuable for factual, grounded question answering.`,
    keyPoints: [
      'Extractive: finds answer spans in context, factually grounded, no hallucination',
      'Generative: creates free-form answers, can synthesize, risks hallucination',
      'Extractive advantages: accuracy, interpretability, speed',
      'Generative advantages: fluency, synthesis, natural responses',
      'Use extractive for factual accuracy, generative for conversational AI',
      'Modern systems often combine both in hybrid approaches',
    ],
  },
  {
    id: 'qa-ir-dq-2',
    question:
      'Compare sparse retrieval (BM25) with dense retrieval (embeddings). Why would you use a hybrid approach?',
    sampleAnswer: `Sparse and dense retrieval represent different philosophies for finding relevant documents:

**Sparse Retrieval (BM25):**

Based on term frequency and inverse document frequency:

**How it works:**
\`\`\`
BM25(Q, D) = Σ IDF(qi) * (f (qi, D) * (k1 + 1)) / 
                         (f (qi, D) + k1 * (1 - b + b * |D| / avgdl))
where:
- f (qi, D): term frequency in document
- |D|: document length
- avgdl: average document length
- k1, b: tuning parameters
\`\`\`

**Characteristics:**
- Exact term matching (lexical matching)
- Sparse vectors (mostly zeros)
- Fast: inverted index lookups
- Interpretable: can see which terms matched

**Strengths:**1. **Exact matching**: Perfect for specific terms, acronyms, IDs
   - Query: "COVID-19" → Must match exactly
2. **Rare terms**: Heavily weights unique terms
3. **No training**: Works out of the box
4. **Transparent**: Can explain why document matched
5. **Fast**: Efficient inverted index
6. **Works with any language**

**Weaknesses:**1. **Vocabulary mismatch**: "car" doesn't match "automobile"
2. **No semantic understanding**: "hot" (temperature) vs "hot" (popular)
3. **Synonyms missed**: "ML" vs "machine learning"
4. **Poor for conceptual queries**: "How to be happy?" has no clear terms

**Dense Retrieval (Semantic Search):**

Based on neural embeddings:

**How it works:**
\`\`\`python
query_embedding = encoder (query)  # [768] dense vector
doc_embeddings = encoder (documents)  # [N, 768]
similarities = cosine_similarity (query_embedding, doc_embeddings)
\`\`\`

**Characteristics:**
- Dense vectors (all non-zero)
- Semantic similarity
- Requires training
- Black box

**Strengths:**1. **Semantic matching**: Understands meaning
   - "car" and "automobile" have similar embeddings
2. **Handles synonyms**: Automatically learned from data
3. **Context-aware**: "Apple" (company) vs "apple" (fruit) distinguished
4. **Cross-lingual**: Can match across languages (multilingual models)
5. **Conceptual queries**: "vacation destinations" matches "beach paradise"

**Weaknesses:**1. **Exact match failures**: May miss specific IDs, codes, acronyms
2. **Computationally expensive**: Vector search over all documents
3. **Requires training**: Needs labeled data or good pre-training
4. **Less interpretable**: Can't easily explain why documents matched
5. **Out-of-distribution**: Struggles with unseen domains

**Hybrid Approach:**

Combines strengths of both:

\`\`\`python
def hybrid_search (query, documents, alpha=0.5):
    # Sparse scores (BM25)
    bm25_scores = bm25.get_scores (query)
    
    # Dense scores (embeddings)
    dense_scores = cosine_similarity (encode (query), encode (documents))
    
    # Normalize to [0, 1]
    bm25_norm = normalize (bm25_scores)
    dense_norm = normalize (dense_scores)
    
    # Weighted combination
    final_scores = alpha * dense_norm + (1 - alpha) * bm25_norm
    
    return final_scores
\`\`\`

**Why Hybrid Works:**

**Example 1: Technical query**
Query: "Python pandas dataframe merge"
- BM25: Strong on exact terms ("pandas", "dataframe", "merge")
- Dense: Understands "join" is similar to "merge"
- Hybrid: Gets both exact matches and semantic variants

**Example 2: Conceptual query**
Query: "How to improve sleep quality?"
- BM25: Weak (no exact terms like "sleep quality" in docs)
- Dense: Strong (matches "better rest", "insomnia solutions")
- Hybrid: Balances both

**Example 3: Specific ID**
Query: "Document #12345"
- BM25: Perfect exact match on "12345"
- Dense: May miss specific number
- Hybrid: BM25 dominates, ensuring exact match

**Practical Implementation:**

\`\`\`python
# Elasticsearch with dense vectors
from elasticsearch import Elasticsearch

# Hybrid query
query = {
    "query": {
        "bool": {
            "should": [
                # Sparse (BM25)
                {"match": {"content": {"query": query_text, "boost": 0.5}}},
                # Dense (embeddings)
                {"script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity (params.query_vector, 'content_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    },
                    "boost": 0.5
                }}
            ]
        }
    }
}
\`\`\`

**When to Use Each:**

**BM25 alone:**
- Exact matching critical (legal, medical records)
- No training data available
- Interpretability required
- Legacy systems

**Dense alone:**
- Semantic understanding critical
- Synonyms and variations common
- Cross-lingual search
- Have good training data

**Hybrid (recommended for most):**
- Best of both worlds
- Production search systems
- Handle diverse query types
- Robust to edge cases

**Tuning α (alpha):**
- α = 0: Pure BM25
- α = 0.3: BM25-heavy (technical docs)
- α = 0.5: Balanced (general purpose)
- α = 0.7: Dense-heavy (conceptual search)
- α = 1: Pure dense

**Modern Trends:**

- **Late interaction** (ColBERT): Per-token embeddings + efficient scoring
- **Sparse-dense hybrids** (SPLADE): Learned sparse representations
- **Query-dependent weighting**: Automatically adjust α per query

**Key Insight:**

Hybrid search provides robustness. BM25 catches what dense misses (exact terms) and dense catches what BM25 misses (semantics). In production, this combination typically outperforms either approach alone by 10-20%.`,
    keyPoints: [
      'BM25 (sparse): exact term matching, fast, interpretable, misses synonyms',
      'Dense (embeddings): semantic matching, handles synonyms, expensive',
      'BM25 strong for exact terms, weak for concepts',
      'Dense strong for concepts, weak for exact matches',
      'Hybrid combines both: α * dense + (1-α) * sparse',
      'Hybrid outperforms either alone by 10-20% in production',
      'Tune α based on domain: technical→lower, conceptual→higher',
    ],
  },
  {
    id: 'qa-ir-dq-3',
    question:
      'Design a production QA system for a customer support chatbot. Describe the architecture, including retrieval, ranking, and answer generation components.',
    sampleAnswer: `A production QA system for customer support requires multiple components working together:

**System Architecture:**

\`\`\`
User Query
    ↓
Query Understanding & Preprocessing
    ↓
Document Retrieval (Hybrid: BM25 + Dense)
    ↓
Re-ranking
    ↓
Answer Extraction/Generation
    ↓
Post-processing & Validation
    ↓
Response with Sources
\`\`\`

**Component 1: Query Understanding**

\`\`\`python
class QueryProcessor:
    def __init__(self):
        self.spell_checker = SpellChecker()
        self.intent_classifier = IntentClassifier()
        
    def process (self, query):
        # Spelling correction
        corrected = self.spell_checker.correct (query)
        
        # Intent classification
        intent = self.intent_classifier.predict (corrected)
        # Intents: "refund", "shipping", "product_info", etc.
        
        # Query expansion (add synonyms)
        expanded = self.expand_query (corrected)
        
        return {
            'original': query,
            'corrected': corrected,
            'expanded': expanded,
            'intent': intent
        }
\`\`\`

**Component 2: Document Store**

\`\`\`python
# Elasticsearch for hybrid search
from elasticsearch import Elasticsearch

class DocumentStore:
    def __init__(self):
        self.es = Elasticsearch()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def index_documents (self, documents):
        """Index with both text and embeddings"""
        for doc in documents:
            embedding = self.encoder.encode (doc['content',])
            
            self.es.index(
                index='support_docs',
                body={
                    'content': doc['content',],
                    'content_embedding': embedding.tolist(),
                    'category': doc['category',],
                    'last_updated': doc['date',]
                }
            )
\`\`\`

**Component 3: Hybrid Retrieval**

\`\`\`python
class HybridRetriever:
    def __init__(self, doc_store):
        self.doc_store = doc_store
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def retrieve (self, query, intent=None, top_k=20):
        ''Retrieve using hybrid search''
        # Encode query
        query_embedding = self.encoder.encode (query)
        
        # Elasticsearch hybrid query
        es_query = {
            "query": {
                "bool": {
                    "should": [
                        # BM25 component
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content",],
                                "type": "best_fields",
                                "boost": 0.5
                            }
                        },
                        # Dense component
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity (params.query_vector, 'content_embedding') + 1.0",
                                    "params": {"query_vector": query_embedding.tolist()}
                                },
                                "boost": 0.5
                            }
                        }
                    ],
                    # Filter by intent/category if available
                    "filter": [
                        {"term": {"category": intent}} if intent else {"match_all": {}}
                    ]
                }
            },
            "size": top_k
        }
        
        results = self.doc_store.es.search (index='support_docs', body=es_query)
        return [hit['_source',] for hit in results['hits',]['hits',]]
\`\`\`

**Component 4: Re-ranking**

\`\`\`python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Reranker:
    def __init__(self):
        # Cross-encoder for re-ranking (more accurate than bi-encoder)
        self.model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def rerank (self, query, documents, top_k=5):
        ''Re-rank with cross-encoder''
        scores = []
        
        for doc in documents:
            inputs = self.tokenizer(
                query, 
                doc['content',],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                score = self.model(**inputs).logits.item()
            
            scores.append (score)
        
        # Sort by score
        ranked_indices = np.argsort (scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices], [scores[i] for i in ranked_indices]
\`\`\`

**Component 5: Answer Extraction**

\`\`\`python
class AnswerExtractor:
    def __init__(self):
        self.qa_model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
        
    def extract_answer (self, question, documents):
        ''Try to extract answer from each document''
        candidates = []
        
        for doc in documents:
            try:
                result = self.qa_model (question=question, context=doc['content',])
                
                # Only include if confidence is high
                if result['score',] > 0.3:
                    candidates.append({
                        'answer': result['answer',],
                        'score': result['score',],
                        'context': doc['content',],
                        'source': doc.get('title', 'Support Document')
                    })
            except:
                continue
        
        # Return best answer
        if candidates:
            return max (candidates, key=lambda x: x['score',])
        return None
\`\`\`

**Component 6: Fallback Generation**

\`\`\`python
class GenerativeAnswerer:
    def __init__(self):
        self.generator = pipeline('text-generation', model='gpt2')
        
    def generate_answer (self, question, context_docs):
        ''Generate answer when extraction fails''
        # Concatenate top documents as context
        context = "\\n".join([doc['content',][:200] for doc in context_docs[:3]])
        
        prompt = f''Based on the following information, answer the question.
        
Context: {context}

Question: {question}

Answer:''
        
        answer = self.generator (prompt, max_length=150)[0]['generated_text',]
        return answer.split("Answer:")[-1].strip()
\`\`\`

**Component 7: Complete System**

\`\`\`python
class CustomerSupportQA:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.retriever = HybridRetriever (doc_store)
        self.reranker = Reranker()
        self.extractor = AnswerExtractor()
        self.generator = GenerativeAnswerer()
        
    def answer_question (self, query):
        # 1. Process query
        processed = self.query_processor.process (query)
        
        # 2. Retrieve candidates (broad recall)
        candidates = self.retriever.retrieve(
            processed['expanded',],
            intent=processed['intent',],
            top_k=20
        )
        
        # 3. Re-rank (precision)
        reranked, scores = self.reranker.rerank (processed['corrected',], candidates, top_k=5)
        
        # 4. Extract answer
        answer = self.extractor.extract_answer (processed['corrected',], reranked)
        
        # 5. Fallback to generation if extraction fails
        if not answer or answer['score',] < 0.5:
            answer = {
                'answer': self.generator.generate_answer (processed['corrected',], reranked),
                'score': 0.7,
                'context': reranked[0]['content',],
                'source': 'Generated from context'
            }
        
        # 6. Format response
        return {
            'answer': answer['answer',],
            'confidence': answer['score',],
            'sources': [
                {'title': doc.get('title', '), 'snippet': doc['content',][:200]}
                for doc in reranked[:3]
            ],
            'intent': processed['intent',]
        }

# Usage
qa_system = CustomerSupportQA()
result = qa_system.answer_question("How do I return a product?")

print(f"Answer: {result['answer',]}")
print(f"Confidence: {result['confidence',]}")
print("Sources:", result['sources',])
\`\`\`

**Production Considerations:**

**1. Caching:**
\`\`\`python
from functools import lru_cache

@lru_cache (maxsize=1000)
def answer_question_cached (query):
    return qa_system.answer_question (query)
\`\`\`

**2. Monitoring:**
\`\`\`python
# Track metrics
- Query latency (p50, p95, p99)
- Cache hit rate
- Confidence scores distribution
- No-answer rate
- User feedback (thumbs up/down)
\`\`\`

**3. Fallbacks:**
\`\`\`python
if confidence < 0.5:
    return "I'm not confident about this answer. Would you like to speak with a human agent?"
\`\`\`

**4. Continuous Improvement:**
- Log all queries and answers
- Collect user feedback
- Retrain on new support docs
- A/B test different models

**5. Safety:**
- Content filtering (no PII in responses)
- Answer validation (no hallucinations)
- Human-in-the-loop for critical queries

**Key Insights:**

- **Multi-stage pipeline**: Recall (retrieval) → Precision (re-ranking) → Answer
- **Hybrid approach**: Combines multiple techniques
- **Graceful degradation**: Extraction → Generation → Human handoff
- **Monitor and iterate**: Track performance, collect feedback, improve`,
    keyPoints: [
      'Multi-stage: query processing → retrieval → re-ranking → extraction',
      'Hybrid retrieval: BM25 + dense embeddings for robustness',
      'Re-ranker (cross-encoder) improves precision after broad retrieval',
      'Fallback chain: extraction → generation → human handoff',
      'Production: caching, monitoring, confidence thresholds',
      'Continuous improvement: log queries, collect feedback, retrain',
    ],
  },
];
