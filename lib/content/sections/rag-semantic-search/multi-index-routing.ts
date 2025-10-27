export const multiIndexRouting = {
  title: 'Multi-Index & Routing',
  content: `
# Multi-Index & Routing

## Introduction

As RAG systems scale, a single index becomes insufficient. Multi-index architectures with intelligent query routing enable specialized retrieval, improved accuracy, and better performance. This is how production systems handle diverse document types and query patterns.

In this comprehensive section, we'll explore index specialization, query classification, routing strategies, and building production multi-index RAG systems.

## Why Multiple Indexes?

Single index limitations:

\`\`\`python
# Problem: One-size-fits-all index
all_documents = [
    "Python tutorial...",           # Code documentation
    "Q3 financial report...",       # Financial data
    "Product specifications...",    # Technical specs
    "Customer support FAQ...",      # Support content
]

# All documents in one index - suboptimal!
single_index.add_all (all_documents)

# Query: "How to use async in Python?"
# Problem: Searches across ALL document types
# - Wastes compute on financial reports
# - May retrieve irrelevant content
# - Slower than necessary
\`\`\`

### Benefits of Multiple Indexes

1. **Specialized Retrieval**: Optimize per document type
2. **Better Accuracy**: Domain-specific embeddings
3. **Faster Queries**: Search only relevant indexes
4. **Cost Efficiency**: Don't search unnecessary indexes
5. **Scalability**: Scale indexes independently

## Index Specialization Strategies

Different ways to split indexes:

### By Document Type

\`\`\`python
from typing import Dict, List
from enum import Enum

class DocumentType(Enum):
    """Document type categories."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    FINANCIAL = "financial"
    SUPPORT = "support"
    GENERAL = "general"

class MultiIndexManager:
    """
    Manage multiple specialized indexes.
    """
    
    def __init__(self):
        self.indexes: Dict[DocumentType, VectorStore] = {}
        self._initialize_indexes()
    
    def _initialize_indexes (self):
        """Create specialized indexes."""
        for doc_type in DocumentType:
            # Each index can have different configuration
            self.indexes[doc_type] = VectorStore(
                name=f"index_{doc_type.value}",
                embedding_model=self._get_embedding_model (doc_type)
            )
    
    def _get_embedding_model (self, doc_type: DocumentType) -> str:
        """Get specialized embedding model for document type."""
        model_map = {
            DocumentType.CODE: "code-embedding-model",
            DocumentType.DOCUMENTATION: "text-embedding-3-small",
            DocumentType.FINANCIAL: "text-embedding-3-large",
            DocumentType.SUPPORT: "text-embedding-3-small",
            DocumentType.GENERAL: "text-embedding-3-small",
        }
        return model_map.get (doc_type, "text-embedding-3-small")
    
    def add_document(
        self,
        document: str,
        doc_type: DocumentType,
        metadata: Dict
    ):
        """
        Add document to appropriate index.
        
        Args:
            document: Document text
            doc_type: Document type
            metadata: Document metadata
        """
        index = self.indexes[doc_type]
        index.add (document, metadata)
    
    def search(
        self,
        query: str,
        doc_types: List[DocumentType],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search across specified indexes.
        
        Args:
            query: Search query
            doc_types: Which indexes to search
            top_k: Results per index
        
        Returns:
            Combined search results
        """
        all_results = []
        
        for doc_type in doc_types:
            index = self.indexes[doc_type]
            results = index.search (query, top_k=top_k)
            
            # Add index info to results
            for result in results:
                result['source_index'] = doc_type.value
            
            all_results.extend (results)
        
        # Re-rank combined results
        all_results.sort (key=lambda x: x['score'], reverse=True)
        
        return all_results[:top_k]


# Example usage
manager = MultiIndexManager()

# Add documents to specialized indexes
manager.add_document(
    "def async_function(): ...",
    DocumentType.CODE,
    {"language": "python"}
)

manager.add_document(
    "Q3 revenue was $10M...",
    DocumentType.FINANCIAL,
    {"quarter": "Q3"}
)

# Search only code index
results = manager.search(
    "async function example",
    doc_types=[DocumentType.CODE],
    top_k=5
)
\`\`\`

### By Data Source

\`\`\`python
class SourceBasedIndexing:
    """
    Organize indexes by data source.
    """
    
    def __init__(self):
        self.indexes = {
            "github": VectorStore("github_index"),
            "confluence": VectorStore("confluence_index"),
            "slack": VectorStore("slack_index"),
            "jira": VectorStore("jira_index"),
        }
    
    def search_by_source(
        self,
        query: str,
        sources: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """Search specific data sources."""
        results = []
        
        for source in sources:
            if source in self.indexes:
                source_results = self.indexes[source].search (query, top_k)
                results.extend (source_results)
        
        return results
\`\`\`

### By Time Period

\`\`\`python
from datetime import datetime, timedelta

class TimeBasedIndexing:
    """
    Organize indexes by time period for efficient temporal queries.
    """
    
    def __init__(self):
        self.indexes = {
            "current_week": VectorStore("current_week"),
            "current_month": VectorStore("current_month"),
            "current_quarter": VectorStore("current_quarter"),
            "historical": VectorStore("historical")
        }
    
    def get_index_for_date (self, date: datetime) -> str:
        """Determine which index to use for a date."""
        now = datetime.now()
        
        if date > now - timedelta (days=7):
            return "current_week"
        elif date > now - timedelta (days=30):
            return "current_month"
        elif date > now - timedelta (days=90):
            return "current_quarter"
        else:
            return "historical"
\`\`\`

## Query Routing Strategies

Intelligent routing to appropriate indexes:

### Rule-Based Routing

\`\`\`python
import re
from typing import List

class RuleBasedRouter:
    """
    Route queries using predefined rules.
    """
    
    def __init__(self):
        self.rules = self._define_rules()
    
    def _define_rules (self) -> List[Dict]:
        """Define routing rules."""
        return [
            {
                "pattern": r"\\b (code|function|class|method|API)\\b",
                "index": DocumentType.CODE,
                "priority": 1
            },
            {
                "pattern": r"\\b (revenue|profit|financial|earnings)\\b",
                "index": DocumentType.FINANCIAL,
                "priority": 1
            },
            {
                "pattern": r"\\b (how to|tutorial|guide|example)\\b",
                "index": DocumentType.DOCUMENTATION,
                "priority": 1
            },
            {
                "pattern": r"\\b (error|issue|problem|help|support)\\b",
                "index": DocumentType.SUPPORT,
                "priority": 1
            },
        ]
    
    def route (self, query: str) -> List[DocumentType]:
        """
        Route query to appropriate indexes.
        
        Args:
            query: User query
        
        Returns:
            List of document types to search
        """
        query_lower = query.lower()
        matched_indexes = []
        
        for rule in self.rules:
            if re.search (rule["pattern"], query_lower, re.IGNORECASE):
                matched_indexes.append (rule["index"])
        
        # If no rules match, search all indexes
        if not matched_indexes:
            matched_indexes = list(DocumentType)
        
        # Remove duplicates
        return list (set (matched_indexes))


# Example usage
router = RuleBasedRouter()

queries = [
    "How to write async functions in Python?",
    "What was Q3 revenue?",
    "I'm getting an error with the API"
]

for query in queries:
    indexes = router.route (query)
    print(f"Query: {query}")
    print(f"Route to: {[idx.value for idx in indexes]}\\n")
\`\`\`

### ML-Based Routing

\`\`\`python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List

class MLQueryRouter:
    """
    Use machine learning to route queries.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer (max_features=1000)
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.index_labels = list(DocumentType)
    
    def train(
        self,
        queries: List[str],
        labels: List[DocumentType]
    ):
        """
        Train the routing classifier.
        
        Args:
            queries: Training queries
            labels: Correct index for each query
        """
        # Vectorize queries
        X = self.vectorizer.fit_transform (queries)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def route(
        self,
        query: str,
        confidence_threshold: float = 0.3
    ) -> List[DocumentType]:
        """
        Route query using trained model.
        
        Args:
            query: User query
            confidence_threshold: Minimum confidence to include index
        
        Returns:
            List of indexes to search
        """
        if not self.is_trained:
            return list(DocumentType)  # Search all if not trained
        
        # Vectorize query
        X = self.vectorizer.transform([query])
        
        # Get probabilities for each index
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Select indexes above threshold
        selected_indexes = [
            self.index_labels[i]
            for i, prob in enumerate (probabilities)
            if prob >= confidence_threshold
        ]
        
        # If none selected, use top prediction
        if not selected_indexes:
            top_index = self.classifier.predict(X)[0]
            selected_indexes = [top_index]
        
        return selected_indexes


# Example training
router = MLQueryRouter()

training_queries = [
    "How to implement async?",
    "Show me code examples",
    "What was the revenue?",
    "Financial performance metrics",
    "I need help with an error",
    "Support for API issues"
]

training_labels = [
    DocumentType.CODE,
    DocumentType.CODE,
    DocumentType.FINANCIAL,
    DocumentType.FINANCIAL,
    DocumentType.SUPPORT,
    DocumentType.SUPPORT,
]

router.train (training_queries, training_labels)

# Route new queries
query = "How do I fix this code error?"
indexes = router.route (query)
print(f"Route '{query}' to: {[idx.value for idx in indexes]}")
\`\`\`

### LLM-Based Routing

\`\`\`python
from openai import OpenAI

client = OpenAI()

class LLMQueryRouter:
    """
    Use LLM to intelligently route queries.
    """
    
    def __init__(self):
        self.client = OpenAI()
    
    def route(
        self,
        query: str,
        available_indexes: Dict[str, str]
    ) -> List[str]:
        """
        Use LLM to determine best indexes.
        
        Args:
            query: User query
            available_indexes: Dict of index_name -> description
        
        Returns:
            List of index names to search
        """
        # Format index descriptions
        index_desc = "\\n".join([
            f"- {name}: {desc}"
            for name, desc in available_indexes.items()
        ])
        
        prompt = f"""Given the user query and available indexes, determine which indexes should be searched.

User Query: {query}

Available Indexes:
{index_desc}

Select the most relevant indexes to search. Return as a comma-separated list.

Selected Indexes:"""
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at routing queries to the most relevant data sources."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        # Parse response
        content = response.choices[0].message.content
        indexes = [idx.strip() for idx in content.split(",")]
        
        # Validate indexes exist
        valid_indexes = [
            idx for idx in indexes
            if idx in available_indexes
        ]
        
        return valid_indexes if valid_indexes else list (available_indexes.keys())


# Example usage
llm_router = LLMQueryRouter()

indexes = {
    "code": "Code examples, API documentation, function definitions",
    "financial": "Financial reports, revenue data, metrics",
    "support": "FAQs, troubleshooting, error solutions"
}

query = "How to calculate quarterly revenue in Python?"
selected = llm_router.route (query, indexes)
print(f"LLM routed to: {selected}")
# Likely: ["financial", "code"]
\`\`\`

## Production Multi-Index System

Complete production implementation:

\`\`\`python
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Search result with metadata."""
    text: str
    score: float
    source_index: str
    metadata: Dict

class ProductionMultiIndexRAG:
    """
    Production-ready multi-index RAG system.
    """
    
    def __init__(
        self,
        routing_strategy: str = "hybrid"
    ):
        """
        Initialize multi-index system.
        
        Args:
            routing_strategy: 'rule', 'ml', 'llm', or 'hybrid'
        """
        self.index_manager = MultiIndexManager()
        
        # Initialize routers
        self.rule_router = RuleBasedRouter()
        self.ml_router = MLQueryRouter()
        self.llm_router = LLMQueryRouter()
        
        self.routing_strategy = routing_strategy
        self.query_log = []
    
    def query(
        self,
        user_query: str,
        max_indexes: int = 3,
        top_k: int = 5
    ) -> Dict:
        """
        Query multi-index system.
        
        Args:
            user_query: User\'s query
            max_indexes: Maximum indexes to search
            top_k: Results per index
        
        Returns:
            Search results with routing info
        """
        # Route query to appropriate indexes
        target_indexes = self._route_query (user_query, max_indexes)
        
        print(f"Routing to indexes: {[idx.value for idx in target_indexes]}")
        
        # Search selected indexes
        results = self.index_manager.search(
            user_query,
            target_indexes,
            top_k=top_k
        )
        
        # Log for monitoring
        self._log_query (user_query, target_indexes, results)
        
        return {
            "results": results,
            "searched_indexes": [idx.value for idx in target_indexes],
            "routing_strategy": self.routing_strategy
        }
    
    def _route_query(
        self,
        query: str,
        max_indexes: int
    ) -> List[DocumentType]:
        """Route query using selected strategy."""
        
        if self.routing_strategy == "rule":
            return self.rule_router.route (query)[:max_indexes]
        
        elif self.routing_strategy == "ml":
            return self.ml_router.route (query)[:max_indexes]
        
        elif self.routing_strategy == "llm":
            # Convert to LLM format
            index_map = {
                idx.value: f"{idx.value} documents"
                for idx in DocumentType
            }
            index_names = self.llm_router.route (query, index_map)
            return [
                DocumentType (name) for name in index_names
                if name in [idx.value for idx in DocumentType]
            ][:max_indexes]
        
        elif self.routing_strategy == "hybrid":
            # Combine multiple strategies
            rule_indexes = set (self.rule_router.route (query))
            ml_indexes = set (self.ml_router.route (query))
            
            # Union of both
            combined = list (rule_indexes | ml_indexes)
            return combined[:max_indexes]
        
        else:
            return list(DocumentType)[:max_indexes]
    
    def _log_query(
        self,
        query: str,
        indexes: List[DocumentType],
        results: List[Dict]
    ):
        """Log query for analysis."""
        self.query_log.append({
            "query": query,
            "indexes": [idx.value for idx in indexes],
            "num_results": len (results),
            "timestamp": datetime.now().isoformat()
        })
    
    def get_routing_stats (self) -> Dict:
        """Get statistics on routing patterns."""
        if not self.query_log:
            return {}
        
        # Count index usage
        index_counts = {}
        for log_entry in self.query_log:
            for index in log_entry["indexes"]:
                index_counts[index] = index_counts.get (index, 0) + 1
        
        return {
            "total_queries": len (self.query_log),
            "index_usage": index_counts,
            "avg_indexes_per_query": np.mean([
                len (log["indexes"]) for log in self.query_log
            ])
        }


# Example usage
rag = ProductionMultiIndexRAG(routing_strategy="hybrid")

# Query the system
result = rag.query(
    "How to implement async error handling in Python?",
    max_indexes=2,
    top_k=5
)

print(f"Searched indexes: {result['searched_indexes']}")
print(f"Found {len (result['results'])} results")

# Get routing statistics
stats = rag.get_routing_stats()
print(f"\\nRouting stats: {stats}")
\`\`\`

## Index Synchronization

Keep multiple indexes in sync:

\`\`\`python
class IndexSynchronizer:
    """
    Synchronize documents across indexes.
    """
    
    def __init__(self, index_manager: MultiIndexManager):
        self.index_manager = index_manager
        self.sync_log = []
    
    def add_document_to_all(
        self,
        document: str,
        metadata: Dict
    ):
        """
        Add document to all relevant indexes.
        
        Args:
            document: Document text
            metadata: Document metadata
        """
        # Determine which indexes should have this document
        target_indexes = self._determine_indexes (document, metadata)
        
        for index_type in target_indexes:
            self.index_manager.add_document(
                document,
                index_type,
                metadata
            )
        
        self._log_sync (document, target_indexes)
    
    def _determine_indexes(
        self,
        document: str,
        metadata: Dict
    ) -> List[DocumentType]:
        """Determine which indexes should contain this document."""
        indexes = []
        
        # Add to type-specific index
        if "type" in metadata:
            indexes.append(DocumentType (metadata["type"]))
        
        # Add to GENERAL index as fallback
        indexes.append(DocumentType.GENERAL)
        
        return indexes
    
    def _log_sync (self, document: str, indexes: List[DocumentType]):
        """Log synchronization."""
        self.sync_log.append({
            "document_id": hash (document),
            "indexes": [idx.value for idx in indexes],
            "timestamp": datetime.now().isoformat()
        })
\`\`\`

## Best Practices

### When to Use Multi-Index

✅ **Use Multiple Indexes When:**
- Different document types with distinct characteristics
- Large scale (> 1M documents)
- Clear query patterns
- Need specialized retrieval per type

❌ **Stick with Single Index When:**
- Small document collection (< 100K)
- Homogeneous content
- Simple use case
- Resource constrained

### Routing Strategy Selection

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Rule-Based** | Clear patterns | Fast, predictable | Inflexible |
| **ML-Based** | Consistent patterns | Adaptive | Needs training data |
| **LLM-Based** | Complex routing | Very flexible | Slower, costs $ |
| **Hybrid** | Production | Best accuracy | More complex |

## Summary

Multi-index architectures enable scalable, specialized RAG:

- **Index Specialization**: By type, source, or time
- **Query Routing**: Rule, ML, or LLM-based
- **Production System**: Hybrid routing with monitoring
- **Synchronization**: Keep indexes consistent
- **Monitoring**: Track routing patterns

**Key Takeaway:** Multi-index systems trade complexity for better accuracy and performance at scale.

**Production Pattern:**1. Start with single index
2. Add specialized indexes as you scale
3. Implement simple rule-based routing first
4. Upgrade to ML/LLM routing with data
5. Monitor routing effectiveness
`,
};
