export const buildingKnowledgeBase = {
  title: 'Building a Knowledge Base',
  content: `
# Building a Knowledge Base

## Introduction

A RAG system is only as good as its knowledge base. Building a production knowledge base requires document ingestion pipelines, preprocessing, metadata extraction, deduplication, versioning, and user interfaces. This section covers the complete lifecycle of building and maintaining a knowledge base.

## Document Ingestion Pipeline

End-to-end pipeline for ingesting documents:

\`\`\`python
from typing import List, Dict, Optional
from pathlib import Path
import hashlib
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Document:
    """Document structure."""
    id: str
    content: str
    metadata: Dict
    source: str
    created_at: datetime
    updated_at: datetime

class DocumentIngestionPipeline:
    """
    Complete pipeline for ingesting documents into knowledge base.
    """
    
    def __init__(
        self,
        vector_store,
        embedding_model
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            vector_store: Vector database for storage
            embedding_model: Model for generating embeddings
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.processed_docs = set()
    
    async def ingest_document(
        self,
        file_path: Path,
        source: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Ingest a single document.
        
        Args:
            file_path: Path to document
            source: Document source identifier
            metadata: Additional metadata
        
        Returns:
            Ingestion result
        """
        try:
            # 1. Load document
            content = self._load_document (file_path)
            
            # 2. Generate document ID
            doc_id = self._generate_doc_id (content, str (file_path))
            
            # 3. Check for duplicates
            if await self._is_duplicate (doc_id):
                return {
                    "status": "skipped",
                    "reason": "duplicate",
                    "doc_id": doc_id
                }
            
            # 4. Preprocess content
            processed_content = await self._preprocess (content)
            
            # 5. Extract metadata
            extracted_metadata = await self._extract_metadata(
                processed_content,
                file_path
            )
            
            # 6. Merge metadata
            full_metadata = {
                **extracted_metadata,
                **(metadata or {}),
                "source": source,
                "file_path": str (file_path),
                "ingested_at": datetime.now().isoformat()
            }
            
            # 7. Chunk document
            chunks = await self._chunk_document (processed_content)
            
            # 8. Generate embeddings
            embeddings = await self._generate_embeddings (chunks)
            
            # 9. Store in vector database
            await self._store_chunks(
                doc_id,
                chunks,
                embeddings,
                full_metadata
            )
            
            # 10. Track processed document
            self.processed_docs.add (doc_id)
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "num_chunks": len (chunks),
                "metadata": full_metadata
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str (e),
                "file_path": str (file_path)
            }
    
    def _load_document (self, file_path: Path) -> str:
        """Load document content."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return file_path.read_text (encoding='utf-8')
        
        elif suffix == '.pdf':
            return self._load_pdf (file_path)
        
        elif suffix in ['.docx', '.doc']:
            return self._load_docx (file_path)
        
        elif suffix == '.md':
            return file_path.read_text (encoding='utf-8')
        
        else:
            raise ValueError (f"Unsupported file type: {suffix}")
    
    def _load_pdf (self, file_path: Path) -> str:
        """Load PDF document."""
        import PyPDF2
        
        text = []
        with open (file_path, 'rb') as f:
            reader = PyPDF2.PdfReader (f)
            for page in reader.pages:
                text.append (page.extract_text())
        
        return "\\n\\n".join (text)
    
    def _load_docx (self, file_path: Path) -> str:
        """Load DOCX document."""
        from docx import Document
        
        doc = Document (file_path)
        return "\\n\\n".join([para.text for para in doc.paragraphs])
    
    def _generate_doc_id (self, content: str, file_path: str) -> str:
        """Generate unique document ID."""
        # Use content hash + file path for uniqueness
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        return f"doc_{content_hash}_{path_hash}"
    
    async def _is_duplicate (self, doc_id: str) -> bool:
        """Check if document already processed."""
        # Check in-memory cache
        if doc_id in self.processed_docs:
            return True
        
        # Check in vector store
        exists = await self.vector_store.document_exists (doc_id)
        return exists
    
    async def _preprocess (self, content: str) -> str:
        """Preprocess document content."""
        # Remove excessive whitespace
        content = " ".join (content.split())
        
        # Remove special characters (optional)
        # content = re.sub (r'[^\\w\\s.,!?-]', ', content)
        
        return content.strip()
    
    async def _extract_metadata(
        self,
        content: str,
        file_path: Path
    ) -> Dict:
        """Extract metadata from document."""
        metadata = {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_bytes": file_path.stat().st_size,
            "content_length": len (content),
        }
        
        # Extract title (first line or filename)
        lines = content.split('\\n')
        if lines:
            metadata["title"] = lines[0][:100]  # First 100 chars
        else:
            metadata["title"] = file_path.stem
        
        # Extract date from content (simple pattern)
        import re
        date_pattern = r'\\d{4}-\\d{2}-\\d{2}'
        dates = re.findall (date_pattern, content)
        if dates:
            metadata["extracted_date"] = dates[0]
        
        return metadata
    
    async def _chunk_document (self, content: str) -> List[str]:
        """Chunk document into smaller pieces."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = splitter.split_text (content)
        return chunks
    
    async def _generate_embeddings (self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks."""
        embeddings = []
        
        for chunk in chunks:
            embedding = await self.embedding_model.embed (chunk)
            embeddings.append (embedding)
        
        return embeddings
    
    async def _store_chunks(
        self,
        doc_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict
    ):
        """Store chunks in vector database."""
        for i, (chunk, embedding) in enumerate (zip (chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len (chunks),
                "doc_id": doc_id
            }
            
            await self.vector_store.add(
                id=chunk_id,
                text=chunk,
                embedding=embedding,
                metadata=chunk_metadata
            )


# Example usage
pipeline = DocumentIngestionPipeline (vector_store, embedding_model)

# Ingest document
result = await pipeline.ingest_document(
    Path("./docs/machine_learning.pdf"),
    source="documentation",
    metadata={"category": "technical", "author": "John Doe"}
)

print(f"Ingestion result: {result['status']}")
print(f"Chunks created: {result['num_chunks']}")
\`\`\`

## Batch Ingestion

Ingest multiple documents efficiently:

\`\`\`python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchIngestionPipeline:
    """
    Batch process multiple documents.
    """
    
    def __init__(
        self,
        ingestion_pipeline: DocumentIngestionPipeline,
        max_workers: int = 5
    ):
        """
        Initialize batch pipeline.
        
        Args:
            ingestion_pipeline: Single document pipeline
            max_workers: Concurrent workers
        """
        self.pipeline = ingestion_pipeline
        self.max_workers = max_workers
    
    async def ingest_directory(
        self,
        directory: Path,
        source: str,
        recursive: bool = True,
        file_patterns: List[str] = None
    ) -> Dict:
        """
        Ingest all documents in directory.
        
        Args:
            directory: Directory path
            source: Source identifier
            recursive: Search subdirectories
            file_patterns: File patterns to include (e.g., ['*.pdf', '*.txt'])
        
        Returns:
            Batch ingestion results
        """
        # Find all files
        files = self._find_files (directory, recursive, file_patterns)
        
        print(f"Found {len (files)} files to ingest")
        
        # Process in batches
        results = await self._process_batch (files, source)
        
        # Summarize results
        summary = self._summarize_results (results)
        
        return summary
    
    def _find_files(
        self,
        directory: Path,
        recursive: bool,
        patterns: Optional[List[str]]
    ) -> List[Path]:
        """Find all matching files."""
        files = []
        
        if patterns is None:
            patterns = ['*.pdf', '*.txt', '*.md', '*.docx']
        
        for pattern in patterns:
            if recursive:
                files.extend (directory.rglob (pattern))
            else:
                files.extend (directory.glob (pattern))
        
        return files
    
    async def _process_batch(
        self,
        files: List[Path],
        source: str
    ) -> List[Dict]:
        """Process files in parallel."""
        # Create tasks
        tasks = [
            self.pipeline.ingest_document (file, source)
            for file in files
        ]
        
        # Process with concurrency limit
        results = []
        for i in range(0, len (tasks), self.max_workers):
            batch = tasks[i:i + self.max_workers]
            batch_results = await asyncio.gather(*batch)
            results.extend (batch_results)
            
            print(f"Processed {min (i + self.max_workers, len (tasks))}/{len (tasks)} files")
        
        return results
    
    def _summarize_results (self, results: List[Dict]) -> Dict:
        """Summarize batch results."""
        total = len (results)
        success = sum(1 for r in results if r["status"] == "success")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")
        
        total_chunks = sum(
            r.get("num_chunks", 0)
            for r in results
            if r["status"] == "success"
        )
        
        return {
            "total_files": total,
            "successful": success,
            "skipped": skipped,
            "errors": errors,
            "total_chunks": total_chunks,
            "success_rate": success / total if total > 0 else 0
        }


# Example usage
batch_pipeline = BatchIngestionPipeline (pipeline, max_workers=5)

# Ingest entire directory
summary = await batch_pipeline.ingest_directory(
    Path("./docs"),
    source="company_docs",
    recursive=True,
    file_patterns=['*.pdf', '*.md']
)

print(f"Ingested {summary['successful']}/{summary['total_files']} files")
print(f"Total chunks: {summary['total_chunks']}")
\`\`\`

## Incremental Updates

Handle document updates efficiently:

\`\`\`python
class IncrementalUpdateManager:
    """
    Manage incremental updates to knowledge base.
    """
    
    def __init__(
        self,
        vector_store,
        ingestion_pipeline: DocumentIngestionPipeline
    ):
        self.vector_store = vector_store
        self.pipeline = ingestion_pipeline
        self.document_versions = {}  # doc_id -> version
    
    async def update_document(
        self,
        file_path: Path,
        source: str
    ) -> Dict:
        """
        Update existing document or add new one.
        
        Args:
            file_path: Path to document
            source: Source identifier
        
        Returns:
            Update result
        """
        # Load and generate ID
        content = self.pipeline._load_document (file_path)
        doc_id = self.pipeline._generate_doc_id (content, str (file_path))
        
        # Check if document exists
        exists = await self.vector_store.document_exists (doc_id)
        
        if exists:
            # Document exists, check if content changed
            old_version = self.document_versions.get (doc_id)
            new_version = hashlib.sha256(content.encode()).hexdigest()
            
            if old_version == new_version:
                return {
                    "status": "unchanged",
                    "doc_id": doc_id
                }
            
            # Content changed, update
            return await self._update_existing(
                doc_id,
                file_path,
                source,
                new_version
            )
        else:
            # New document, ingest
            result = await self.pipeline.ingest_document(
                file_path,
                source
            )
            
            if result["status"] == "success":
                version = hashlib.sha256(content.encode()).hexdigest()
                self.document_versions[doc_id] = version
            
            return result
    
    async def _update_existing(
        self,
        doc_id: str,
        file_path: Path,
        source: str,
        new_version: str
    ) -> Dict:
        """Update existing document."""
        # 1. Delete old chunks
        await self.vector_store.delete_by_doc_id (doc_id)
        
        # 2. Re-ingest
        result = await self.pipeline.ingest_document(
            file_path,
            source
        )
        
        # 3. Update version
        if result["status"] == "success":
            self.document_versions[doc_id] = new_version
        
        return {
            **result,
            "status": "updated"
        }
    
    async def delete_document (self, doc_id: str):
        """Delete document and all its chunks."""
        await self.vector_store.delete_by_doc_id (doc_id)
        self.document_versions.pop (doc_id, None)


# Example usage
update_manager = IncrementalUpdateManager (vector_store, pipeline)

# Update document (will skip if unchanged)
result = await update_manager.update_document(
    Path("./docs/ml_guide.pdf"),
    source="docs"
)

print(f"Update status: {result['status']}")
\`\`\`

## Deduplication

Prevent duplicate content:

\`\`\`python
from typing import Set
import numpy as np

class DeduplicationManager:
    """
    Detect and handle duplicate content.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize deduplication manager.
        
        Args:
            similarity_threshold: Threshold for considering duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.content_hashes: Set[str] = set()
        self.embeddings_cache = []
    
    def is_duplicate_by_hash (self, content: str) -> bool:
        """
        Check if content is duplicate using hash.
        
        Args:
            content: Document content
        
        Returns:
            True if duplicate
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        if content_hash in self.content_hashes:
            return True
        
        self.content_hashes.add (content_hash)
        return False
    
    async def is_duplicate_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if content is duplicate using embedding similarity.
        
        Args:
            embedding: Document embedding
            threshold: Similarity threshold
        
        Returns:
            True if duplicate
        """
        threshold = threshold or self.similarity_threshold
        
        if not self.embeddings_cache:
            self.embeddings_cache.append (embedding)
            return False
        
        # Compare with existing embeddings
        for cached_embedding in self.embeddings_cache:
            similarity = self._cosine_similarity (embedding, cached_embedding)
            
            if similarity >= threshold:
                return True
        
        # Not a duplicate, cache it
        self.embeddings_cache.append (embedding)
        return False
    
    def _cosine_similarity(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> float:
        """Calculate cosine similarity."""
        return np.dot (v1, v2) / (np.linalg.norm (v1) * np.linalg.norm (v2))
    
    def find_near_duplicates(
        self,
        documents: List[Dict],
        threshold: float = 0.9
    ) -> List[List[int]]:
        """
        Find near-duplicate documents.
        
        Args:
            documents: List of documents with embeddings
            threshold: Similarity threshold
        
        Returns:
            Groups of duplicate document indices
        """
        n = len (documents)
        duplicates = []
        visited = set()
        
        for i in range (n):
            if i in visited:
                continue
            
            group = [i]
            visited.add (i)
            
            for j in range (i + 1, n):
                if j in visited:
                    continue
                
                similarity = self._cosine_similarity(
                    documents[i]["embedding"],
                    documents[j]["embedding"]
                )
                
                if similarity >= threshold:
                    group.append (j)
                    visited.add (j)
            
            if len (group) > 1:
                duplicates.append (group)
        
        return duplicates


# Example usage
dedup = DeduplicationManager (similarity_threshold=0.95)

# Check for duplicates
content = "This is a document about machine learning."

if dedup.is_duplicate_by_hash (content):
    print("Duplicate detected by hash!")
else:
    print("Not a duplicate")
\`\`\`

## Version Control for Documents

Track document versions:

\`\`\`python
from typing import List
import json

class DocumentVersionControl:
    """
    Version control for knowledge base documents.
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize version control.
        
        Args:
            storage_path: Path to store version history
        """
        self.storage_path = storage_path
        self.storage_path.mkdir (exist_ok=True)
    
    def save_version(
        self,
        doc_id: str,
        content: str,
        metadata: Dict
    ) -> str:
        """
        Save new version of document.
        
        Args:
            doc_id: Document ID
            content: Document content
            metadata: Document metadata
        
        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.now().isoformat()
        version_id = f"{doc_id}_v_{timestamp}"
        
        # Save version
        version_data = {
            "version_id": version_id,
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata,
            "created_at": timestamp
        }
        
        version_file = self.storage_path / f"{version_id}.json"
        with open (version_file, 'w') as f:
            json.dump (version_data, f, indent=2)
        
        return version_id
    
    def get_version (self, version_id: str) -> Optional[Dict]:
        """Get specific version."""
        version_file = self.storage_path / f"{version_id}.json"
        
        if not version_file.exists():
            return None
        
        with open (version_file, 'r') as f:
            return json.load (f)
    
    def list_versions (self, doc_id: str) -> List[Dict]:
        """List all versions of document."""
        versions = []
        
        for version_file in self.storage_path.glob (f"{doc_id}_v_*.json"):
            with open (version_file, 'r') as f:
                versions.append (json.load (f))
        
        # Sort by creation time
        versions.sort (key=lambda v: v["created_at"], reverse=True)
        
        return versions
    
    def rollback_to_version(
        self,
        version_id: str
    ) -> Dict:
        """Rollback to specific version."""
        version_data = self.get_version (version_id)
        
        if not version_data:
            raise ValueError (f"Version not found: {version_id}")
        
        return {
            "doc_id": version_data["doc_id"],
            "content": version_data["content"],
            "metadata": version_data["metadata"]
        }


# Example usage
version_control = DocumentVersionControl(Path("./versions"))

# Save version
version_id = version_control.save_version(
    doc_id="doc_123",
    content="Updated content...",
    metadata={"author": "John", "version": "2.0"}
)

# List versions
versions = version_control.list_versions("doc_123")
print(f"Found {len (versions)} versions")

# Rollback
old_version = version_control.rollback_to_version (version_id)
\`\`\`

## Knowledge Base Management Interface

User interface for managing knowledge base:

\`\`\`python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

# Initialize components
pipeline = DocumentIngestionPipeline (vector_store, embedding_model)
batch_pipeline = BatchIngestionPipeline (pipeline)
update_manager = IncrementalUpdateManager (vector_store, pipeline)

@app.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    source: str = "upload",
    metadata: Optional[str] = None
):
    """
    Upload and ingest document.
    
    Args:
        file: Document file
        source: Source identifier
        metadata: JSON metadata
    """
    # Save uploaded file
    temp_path = Path (f"/tmp/{file.filename}")
    with open (temp_path, 'wb') as f:
        f.write (await file.read())
    
    # Parse metadata
    meta_dict = json.loads (metadata) if metadata else {}
    
    # Ingest
    result = await pipeline.ingest_document(
        temp_path,
        source,
        meta_dict
    )
    
    # Clean up
    temp_path.unlink()
    
    return JSONResponse (result)

@app.post("/api/ingest-batch")
async def ingest_batch(
    directory: str,
    source: str,
    recursive: bool = True
):
    """
    Ingest all documents in directory.
    
    Args:
        directory: Directory path
        source: Source identifier
        recursive: Search subdirectories
    """
    summary = await batch_pipeline.ingest_directory(
        Path (directory),
        source,
        recursive
    )
    
    return JSONResponse (summary)

@app.put("/api/update/{doc_id}")
async def update_document(
    doc_id: str,
    file: UploadFile = File(...)
):
    """
    Update existing document.
    
    Args:
        doc_id: Document ID
        file: New document file
    """
    temp_path = Path (f"/tmp/{file.filename}")
    with open (temp_path, 'wb') as f:
        f.write (await file.read())
    
    result = await update_manager.update_document(
        temp_path,
        source="upload"
    )
    
    temp_path.unlink()
    
    return JSONResponse (result)

@app.delete("/api/documents/{doc_id}")
async def delete_document (doc_id: str):
    """
    Delete document.
    
    Args:
        doc_id: Document ID
    """
    await update_manager.delete_document (doc_id)
    
    return JSONResponse({
        "status": "deleted",
        "doc_id": doc_id
    })

@app.get("/api/documents")
async def list_documents(
    limit: int = 100,
    offset: int = 0
):
    """
    List all documents.
    
    Args:
        limit: Number of documents
        offset: Pagination offset
    """
    docs = await vector_store.list_documents (limit, offset)
    
    return JSONResponse({
        "documents": docs,
        "total": len (docs)
    })

@app.get("/api/stats")
async def get_stats():
    """Get knowledge base statistics."""
    stats = await vector_store.get_stats()
    
    return JSONResponse (stats)
\`\`\`

## Best Practices

### Knowledge Base Checklist

✅ **Ingestion**
- Support multiple file formats
- Extract comprehensive metadata
- Implement deduplication
- Use batch processing for efficiency

✅ **Updates**
- Handle incremental updates
- Version control documents
- Validate before ingesting
- Monitor ingestion pipeline

✅ **Quality**
- Preprocess documents consistently
- Extract meaningful metadata
- Remove duplicates
- Validate content quality

✅ **Maintenance**
- Regular cleanup of old documents
- Monitor storage usage
- Track ingestion failures
- Audit document quality

## Summary

Building a production knowledge base requires:

- **Ingestion Pipeline**: Load, process, chunk, embed, store
- **Batch Processing**: Efficiently handle multiple documents
- **Incremental Updates**: Update without full re-ingestion
- **Deduplication**: Prevent duplicate content
- **Version Control**: Track document history
- **Management Interface**: UI for managing documents

**Key Takeaway:** A well-built knowledge base is the foundation of effective RAG.

**Production Pattern:**
1. Start with simple ingestion
2. Add metadata extraction
3. Implement deduplication
4. Add version control
5. Build management interface
6. Monitor and maintain quality
`,
};
