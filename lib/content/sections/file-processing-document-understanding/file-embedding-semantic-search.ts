/**
 * File Embedding & Semantic Search Section
 * Module 3: File Processing & Document Understanding
 */

export const fileembeddingsemanticsearchSection = {
    id: 'file-embedding-semantic-search',
    title: 'File Embedding & Semantic Search',
    content: `# File Embedding & Semantic Search

Master semantic search across files for building intelligent code search and document retrieval systems like Cursor.

## Creating File Embeddings

\`\`\`python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_text(text: str) -> list:
    """Generate embedding for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def embed_file(filepath: str) -> list:
    """Generate embedding for file content."""
    with open(filepath, 'r') as f:
        content = f.read()
    return embed_text(content)
\`\`\`

## Semantic Search Implementation

\`\`\`python
import numpy as np
from pathlib import Path
from typing import List, Tuple

class FileSemanticSearch:
    """Semantic search across files - like Cursor's search."""
    
    def __init__(self):
        self.client = OpenAI()
        self.file_embeddings = {}
    
    def index_directory(self, dir_path: str, extensions: list = ['.py', '.js', '.ts']):
        """Index all files in directory."""
        for filepath in Path(dir_path).rglob('*'):
            if filepath.suffix in extensions and filepath.is_file():
                try:
                    content = filepath.read_text()
                    embedding = self.embed_text(content)
                    self.file_embeddings[str(filepath)] = {
                        'embedding': embedding,
                        'content': content
                    }
                except:
                    pass
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search files semantically."""
        query_embedding = self.embed_text(query)
        
        results = []
        for filepath, data in self.file_embeddings.items():
            similarity = self.cosine_similarity(
                query_embedding,
                data['embedding']
            )
            results.append((filepath, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def embed_text(self, text: str) -> list:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]  # Truncate if too long
        )
        return response.data[0].embedding
    
    def cosine_similarity(self, a: list, b: list) -> float:
        """Calculate cosine similarity."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Usage - like Cursor's semantic search
searcher = FileSemanticSearch()
searcher.index_directory('src')

results = searcher.search("authentication logic")
for filepath, score in results:
    print(f"{filepath}: {score:.3f}")
\`\`\`

## Vector Database Integration

\`\`\`python
# pip install chromadb
import chromadb
from pathlib import Path

class VectorFileStore:
    """Store file embeddings in vector database."""
    
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("files")
        self.openai_client = OpenAI()
    
    def add_file(self, filepath: str):
        """Add file to vector store."""
        content = Path(filepath).read_text()
        
        # Generate embedding
        embedding = self.embed_text(content)
        
        # Add to collection
        self.collection.add(
            ids=[filepath],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"path": filepath}]
        )
    
    def search(self, query: str, n_results: int = 5):
        """Search files semantically."""
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def embed_text(self, text: str) -> list:
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        return response.data[0].embedding

# Usage
store = VectorFileStore()

# Index files
for file in Path('src').rglob('*.py'):
    store.add_file(str(file))

# Search
results = store.search("database connection logic")
\`\`\`

## Key Takeaways

1. **Use embeddings** for semantic understanding
2. **OpenAI embeddings** are cost-effective
3. **Cosine similarity** for comparing embeddings
4. **Vector databases** for scale (Chroma, Pinecone)
5. **Index file content** not just names
6. **Chunk large files** before embedding
7. **Cache embeddings** to save costs
8. **Update incrementally** when files change
9. **Combine semantic + keyword** search
10. **Use for code search** like Cursor does`,
    videoUrl: null,
};

