export const multiModalRag = {
  title: 'Multi-Modal RAG',
  id: 'multi-modal-rag',
  description:
    'Master building retrieval-augmented generation systems that work across text, images, and other modalities for powerful search and question answering.',
  content: `
# Multi-Modal RAG

## Introduction

Retrieval-Augmented Generation (RAG) is a powerful pattern that combines document retrieval with LLM generation. While traditional RAG focuses on text documents, multi-modal RAG extends this capability to images, videos, audio, and mixed-content documents.

In this section, we'll explore how to build RAG systems that can retrieve and reason over multiple modalities, enabling applications like visual question answering over document collections, image search with natural language, and comprehensive document understanding.

## Why Multi-Modal RAG?

Traditional text-only RAG has limitations:

**Text-Only RAG:**
- Can't access information in images
- Misses charts, diagrams, and visualizations
- Loses spatial and visual context
- Limited to textual descriptions

**Multi-Modal RAG:**
- Retrieves images, videos, and documents
- Understands visual content
- Reasons across text and images together
- Preserves context and relationships

**Use Cases:**
- Technical documentation with diagrams
- Medical records with scans and images
- Product catalogs with photos
- Research papers with figures and tables
- Presentations with slides
- Educational content with illustrations

## Architecture Patterns

### 1. Late Fusion RAG

Retrieve and process modalities separately, then combine.

\`\`\`
User Query → [Text Retrieval] → [Text Documents]
          → [Image Retrieval] → [Images]
          
[Text Documents] + [Images] → [LLM] → Response
\`\`\`

**Pros:**
- Simple to implement
- Can use existing retrieval systems
- Flexible

**Cons:**
- No cross-modal retrieval
- Separate embedding spaces
- May miss relationships

### 2. Joint Embedding RAG

Embed all modalities in shared space for unified retrieval.

\`\`\`
[CLIP Embeddings]
Text + Images → [Shared Vector Space] → [Retrieve Top-K] → [LLM] → Response
\`\`\`

**Pros:**
- Cross-modal retrieval (text query → image results)
- Unified system
- Semantic alignment

**Cons:**
- More complex implementation
- Requires multi-modal embeddings (CLIP, etc.)
- Single embedding may not capture all nuances

### 3. Hierarchical RAG

First retrieve documents, then retrieve relevant modalities within documents.

\`\`\`
Query → [Retrieve Documents] → [Document A, Document B]
     → [Extract relevant images/sections from each] → [LLM] → Response
\`\`\`

**Pros:**
- Efficient
- Context-aware retrieval
- Good for structured documents

**Cons:**
- Two-stage process
- May miss cross-document relationships

## Building Multi-Modal RAG with CLIP

### Setup

\`\`\`python
import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
import clip
import torch
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class MultiModalDocument:
    """Represents a multi-modal document."""
    id: str
    text_content: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None

class MultiModalRAG:
    """Multi-modal RAG system using CLIP embeddings."""
    
    def __init__(
        self,
        openai_api_key: str,
        clip_model: str = "ViT-B/32",
        dimension: int = 512
    ):
        self.client = OpenAI(api_key=openai_api_key)
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # Initialize FAISS index
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Store documents
        self.documents: List[MultiModalDocument] = []
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using CLIP."""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()[0]
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed image using CLIP."""
        image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def add_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a document to the index.
        
        Can be text-only, image-only, or both.
        """
        if not text and not image_path:
            raise ValueError("Must provide either text or image")
        
        # Create embedding
        if text and image_path:
            # Average text and image embeddings
            text_emb = self.embed_text(text)
            image_emb = self.embed_image(image_path)
            embedding = (text_emb + image_emb) / 2
        elif text:
            embedding = self.embed_text(text)
        else:  # image_path
            embedding = self.embed_image(image_path)
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        # Create document
        doc = MultiModalDocument(
            id=doc_id,
            text_content=text,
            image_path=image_path,
            metadata=metadata or {},
            embedding=embedding
        )
        
        # Add to index
        self.index.add(np.array([embedding]).astype('float32'))
        self.documents.append(doc)
    
    def retrieve(
        self,
        query: Union[str, Image.Image],
        top_k: int = 5
    ) -> List[MultiModalDocument]:
        """
        Retrieve documents based on query.
        
        Query can be text or image.
        """
        # Embed query
        if isinstance(query, str):
            query_embedding = self.embed_text(query)
        else:
            # Assume PIL Image
            # Would need to adapt embed_image to accept PIL Image directly
            raise NotImplementedError("Image query not yet implemented")
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            top_k
        )
        
        # Return documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        
        return retrieved_docs
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        include_images_in_response: bool = True
    ) -> str:
        """
        Answer a question using retrieved documents.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            include_images_in_response: Whether to send images to LLM
        
        Returns:
            Answer to the question
        """
        # Retrieve relevant documents
        docs = self.retrieve(question, top_k)
        
        # Build context
        text_context = []
        images_to_include = []
        
        for i, doc in enumerate(docs):
            if doc.text_content:
                text_context.append(f"Document {i+1}:\\n{doc.text_content}")
            
            if doc.image_path and include_images_in_response:
                images_to_include.append(doc.image_path)
        
        # Prepare messages
        if include_images_in_response and images_to_include:
            # Multi-modal response with images
            content = [
                {
                    "type": "text",
                    "text": f"""Answer this question using the provided documents and images:

Question: {question}

Text Documents:
{chr(10).join(text_context)}

Analyze the images provided and incorporate any relevant information from them in your answer."""
                }
            ]
            
            # Add images
            import base64
            for img_path in images_to_include[:5]:  # Limit to 5 images
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                            "detail": "low"
                        }
                    })
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
        else:
            # Text-only response
            prompt = f"""Answer this question using the provided context:

Question: {question}

Context:
{chr(10).join(text_context)}

Provide a clear and concise answer based on the context."""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
        
        return response.choices[0].message.content

# Example usage
rag = MultiModalRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Add documents
rag.add_document(
    "doc1",
    text="This chart shows Q1 revenue of $1.2M, up 15% from Q4.",
    image_path="revenue_chart.png",
    metadata={"type": "financial", "quarter": "Q1"}
)

rag.add_document(
    "doc2",
    text="Product roadmap for 2024 includes three major releases.",
    image_path="roadmap.png",
    metadata={"type": "planning", "year": 2024}
)

rag.add_document(
    "doc3",
    text="Customer satisfaction scores improved to 4.5/5 stars.",
    metadata={"type": "metrics"}
)

# Query the system
answer = rag.query("What was our Q1 revenue?", top_k=3)
print(answer)
\`\`\`

## PDF Processing with Images

\`\`\`python
import fitz  # PyMuPDF
from typing import List, Tuple
import io

def extract_pdf_content(
    pdf_path: str
) -> List[Dict[str, Any]]:
    """
    Extract text and images from PDF.
    
    Returns list of page contents with text and images.
    """
    doc = fitz.open(pdf_path)
    
    contents = []
    
    for page_num, page in enumerate(doc):
        page_content = {
            "page": page_num + 1,
            "text": page.get_text(),
            "images": []
        }
        
        # Extract images
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image temporarily
            image_path = f"page_{page_num+1}_img_{img_index}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            page_content["images"].append(image_path)
        
        contents.append(page_content)
    
    doc.close()
    
    return contents

def index_pdf(
    rag: MultiModalRAG,
    pdf_path: str,
    doc_prefix: str
):
    """Index a PDF file into multi-modal RAG."""
    contents = extract_pdf_content(pdf_path)
    
    for page_content in contents:
        page_num = page_content["page"]
        text = page_content["text"]
        images = page_content["images"]
        
        # Index page text
        if text.strip():
            rag.add_document(
                doc_id=f"{doc_prefix}_page_{page_num}",
                text=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "type": "pdf_text"
                }
            )
        
        # Index each image with surrounding text as context
        for img_idx, img_path in enumerate(images):
            rag.add_document(
                doc_id=f"{doc_prefix}_page_{page_num}_img_{img_idx}",
                text=text,  # Use page text as context
                image_path=img_path,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "image_index": img_idx,
                    "type": "pdf_image"
                }
            )

# Index multiple PDFs
rag = MultiModalRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))

index_pdf(rag, "annual_report.pdf", "report_2023")
index_pdf(rag, "product_guide.pdf", "product")

# Query across all documents
answer = rag.query("What were the key product features mentioned?")
print(answer)
\`\`\`

## Multi-Modal RAG with LangChain

\`\`\`python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import base64

class LangChainMultiModalRAG:
    """Multi-modal RAG using LangChain."""
    
    def __init__(
        self,
        openai_api_key: str,
        persist_directory: str = "./chroma_db"
    ):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4-vision-preview",
            openai_api_key=openai_api_key
        )
        
        # Store image paths separately
        self.image_registry = {}
    
    def add_text_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add text document to vector store."""
        doc = Document(
            page_content=text,
            metadata=metadata or {}
        )
        
        self.vectorstore.add_documents([doc])
    
    def add_image_with_caption(
        self,
        image_path: str,
        caption: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add image with text caption to index.
        
        The caption is embedded for retrieval, and image path is stored.
        """
        doc_id = f"img_{len(self.image_registry)}"
        
        # Store image path
        self.image_registry[doc_id] = image_path
        
        # Add caption to vector store
        doc = Document(
            page_content=caption,
            metadata={
                **(metadata or {}),
                "doc_id": doc_id,
                "has_image": True,
                "image_path": image_path
            }
        )
        
        self.vectorstore.add_documents([doc])
    
    def query(
        self,
        question: str,
        k: int = 5
    ) -> str:
        """Query the multi-modal knowledge base."""
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # Separate text and image documents
        text_docs = []
        image_docs = []
        
        for doc in docs:
            if doc.metadata.get("has_image"):
                image_docs.append(doc)
            else:
                text_docs.append(doc)
        
        # Build prompt with context
        context_parts = []
        
        for doc in text_docs:
            context_parts.append(doc.page_content)
        
        context = "\\n\\n".join(context_parts)
        
        # If we have images, use vision model
        if image_docs:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Build message with images
            content = [{
                "type": "text",
                "text": f"""Answer this question using the provided context and images:

Question: {question}

Text Context:
{context}

Analyze the images and incorporate relevant visual information in your answer."""
            }]
            
            # Add images
            for img_doc in image_docs[:3]:  # Limit to 3 images
                img_path = img_doc.metadata["image_path"]
                
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}",
                            "detail": "low"
                        }
                    })
            
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        else:
            # Text-only query
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4"),
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
                return_source_documents=False
            )
            
            return qa_chain.run(question)

# Example usage
mm_rag = LangChainMultiModalRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Add text documents
mm_rag.add_text_document(
    "Our Q1 revenue was $1.2M, representing 15% growth.",
    metadata={"type": "financial", "quarter": "Q1"}
)

# Add images with captions
mm_rag.add_image_with_caption(
    "chart.png",
    "Revenue growth chart showing quarterly performance",
    metadata={"type": "visualization"}
)

# Query
answer = mm_rag.query("Show me information about Q1 revenue")
print(answer)
\`\`\`

## Image Captioning for RAG

Automatically generate captions for images to enable text-based retrieval:

\`\`\`python
def generate_image_caption_for_rag(
    image_path: str,
    context: Optional[str] = None
) -> str:
    """
    Generate descriptive caption for image to enable RAG retrieval.
    
    Args:
        image_path: Path to image
        context: Optional textual context (e.g., surrounding paragraph)
    
    Returns:
        Detailed caption
    """
    prompt = """Generate a detailed, searchable description of this image for a document search system.

Include:
- Main subject and objects
- Key visual elements
- Any text visible in the image
- Type of visualization (if chart/diagram)
- Data or information conveyed
- Context and purpose

Be specific and use keywords that would help in search."""

    if context:
        prompt += f"\\n\\nSurrounding text context:\\n{context}"
    
    import base64
    from openai import OpenAI
    
    client = OpenAI()
    
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_data}",
                        "detail": "high"
                    }
                }
            ]
        }],
        max_tokens=300
    )
    
    return response.choices[0].message.content

# Generate caption
caption = generate_image_caption_for_rag(
    "product_diagram.png",
    context="This diagram shows the architecture of our new system."
)
print(caption)

# Add to RAG with caption
rag.add_document(
    "diagram1",
    text=caption,
    image_path="product_diagram.png",
    metadata={"type": "diagram"}
)
\`\`\`

## Hybrid Search

Combine dense (vector) and sparse (keyword) search for better results:

\`\`\`python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridMultiModalRAG:
    """Multi-modal RAG with hybrid search."""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        
        # CLIP for dense retrieval
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # FAISS for vector search
        self.index = faiss.IndexFlatIP(512)
        
        # Documents
        self.documents = []
        
        # BM25 for keyword search
        self.bm25 = None
        self.tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def add_document(
        self,
        doc_id: str,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Add document to both vector and keyword indexes."""
        # Create embedding (same as before)
        if text and image_path:
            text_emb = self._embed_text(text)
            image_emb = self._embed_image(image_path)
            embedding = (text_emb + image_emb) / 2
        elif text:
            embedding = self._embed_text(text)
        else:
            embedding = self._embed_image(image_path)
        
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add to vector index
        self.index.add(np.array([embedding]).astype('float32'))
        
        # Create document
        doc = MultiModalDocument(
            id=doc_id,
            text_content=text or "",
            image_path=image_path,
            metadata=metadata or {},
            embedding=embedding
        )
        
        self.documents.append(doc)
        
        # Add to keyword index
        if text:
            self.tokenized_docs.append(self._tokenize(text))
        else:
            self.tokenized_docs.append([])
        
        # Rebuild BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5  # Weight for vector vs keyword (0 = all keyword, 1 = all vector)
    ) -> List[MultiModalDocument]:
        """
        Retrieve using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results
            alpha: Interpolation between vector (alpha) and keyword (1-alpha)
        
        Returns:
            Top-k documents
        """
        # Vector search
        query_embedding = self._embed_text(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        vector_distances, vector_indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            len(self.documents)
        )
        
        # Normalize vector scores to [0, 1]
        vector_scores = (vector_distances[0] + 1) / 2  # Cosine similarity to [0, 1]
        
        # Keyword search
        tokenized_query = self._tokenize(query)
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize keyword scores
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()
        
        # Combine scores
        combined_scores = alpha * vector_scores + (1 - alpha) * keyword_scores
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        return [self.documents[idx] for idx in top_indices]
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed text with CLIP."""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            features = self.clip_model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]
    
    def _embed_image(self, image_path: str) -> np.ndarray:
        """Embed image with CLIP."""
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

# Usage
hybrid_rag = HybridMultiModalRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Add documents
hybrid_rag.add_document("doc1", text="Machine learning models for image classification")
hybrid_rag.add_document("doc2", text="Deep learning architecture diagrams", image_path="arch.png")

# Hybrid search
results = hybrid_rag.hybrid_retrieve("neural network architectures", top_k=5, alpha=0.7)

for doc in results:
    print(f"- {doc.text_content}")
\`\`\`

## Best Practices

### 1. Image Preprocessing

\`\`\`python
def preprocess_image_for_rag(
    image_path: str,
    max_size: int = 1024
) -> str:
    """Preprocess image for efficient RAG."""
    from PIL import Image
    
    img = Image.open(image_path)
    
    # Resize if too large
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save optimized version
    output_path = f"optimized_{os.path.basename(image_path)}"
    img.save(output_path, 'JPEG', quality=85, optimize=True)
    
    return output_path
\`\`\`

### 2. Chunking Strategies

\`\`\`python
def chunk_document_with_images(
    text: str,
    images: List[str],
    chunk_size: int = 500
) -> List[Dict[str, Any]]:
    """
    Chunk document while preserving image associations.
    
    Returns chunks with associated images.
    """
    # Split text into sentences
    sentences = text.split('. ')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save chunk
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                "text": chunk_text,
                "images": []  # Associate images based on position
            })
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        chunks.append({
            "text": chunk_text,
            "images": []
        })
    
    # Associate images with chunks (simplified - in practice, use page/position info)
    images_per_chunk = len(images) // len(chunks) if chunks else 0
    for i, chunk in enumerate(chunks):
        start_img = i * images_per_chunk
        end_img = start_img + images_per_chunk
        chunk["images"] = images[start_img:end_img]
    
    return chunks
\`\`\`

### 3. Re-ranking

\`\`\`python
def rerank_multimodal_results(
    query: str,
    documents: List[MultiModalDocument],
    top_k: int = 5
) -> List[MultiModalDocument]:
    """
    Re-rank retrieved documents using cross-encoder.
    
    This provides more accurate ranking than initial retrieval.
    """
    from sentence_transformers import CrossEncoder
    
    # Load cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Score each document
    scores = []
    for doc in documents:
        if doc.text_content:
            score = cross_encoder.predict([(query, doc.text_content)])[0]
            scores.append((score, doc))
    
    # Sort by score
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # Return top-k
    return [doc for _, doc in scores[:top_k]]
\`\`\`

## Production Considerations

### Caching

\`\`\`python
import redis
import hashlib
import json

class CachedMultiModalRAG:
    """Multi-modal RAG with caching."""
    
    def __init__(self, openai_api_key: str, redis_host: str = "localhost"):
        self.rag = MultiModalRAG(openai_api_key)
        self.redis_client = redis.Redis(host=redis_host)
        self.cache_ttl = 86400  # 24 hours
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query with caching."""
        # Generate cache key
        cache_key = f"mmrag:{hashlib.sha256(question.encode()).hexdigest()}:{top_k}"
        
        # Check cache
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Query RAG
        answer = self.rag.query(question, top_k)
        
        # Cache result
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(answer))
        
        return answer
\`\`\`

### Monitoring

\`\`\`python
import logging
from datetime import datetime

def log_rag_query(
    query: str,
    num_retrieved: int,
    response_time: float,
    has_images: bool
):
    """Log RAG queries for monitoring."""
    logging.info(
        f"RAG Query: query_length={len(query)}, "
        f"retrieved={num_retrieved}, "
        f"time={response_time:.2f}s, "
        f"has_images={has_images}"
    )
\`\`\`

## Real-World Applications

### 1. Technical Documentation

\`\`\`python
def build_technical_docs_rag(doc_directory: str) -> MultiModalRAG:
    """Build RAG system for technical documentation."""
    rag = MultiModalRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Process all markdown files
    for md_file in Path(doc_directory).rglob("*.md"):
        with open(md_file, 'r') as f:
            content = f.read()
        
        # Find associated images
        images = []
        image_dir = md_file.parent / "images"
        if image_dir.exists():
            images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        # Add to RAG
        for img in images:
            caption = generate_image_caption_for_rag(str(img), context=content[:500])
            rag.add_document(
                f"{md_file.stem}_{img.stem}",
                text=caption,
                image_path=str(img),
                metadata={"source": str(md_file)}
            )
    
    return rag
\`\`\`

### 2. Product Catalog

\`\`\`python
def build_product_catalog_rag(products: List[Dict]) -> MultiModalRAG:
    """Build RAG for product catalog with images."""
    rag = MultiModalRAG(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    for product in products:
        # Combine product information
        text = f"""
        Product: {product['name']}
        Price: \${product['price']}
        Description: {product['description']}
        Features: {', '.join(product['features'])}
        """
        
        # Add to RAG
        rag.add_document(
            f"product_{product['id']}",
            text=text,
            image_path=product['image_path'],
            metadata={
                "category": product['category'],
                "price": product['price']
            }
        )
    
    return rag

# Query product catalog
answer = rag.query("Show me wireless headphones under $100")
\`\`\`

## Summary

Multi-modal RAG enables powerful retrieval and generation across text, images, and other modalities:

**Key Capabilities:**
- Cross-modal retrieval (text query → image results)
- Joint reasoning over text and images
- PDF processing with embedded images
- Hybrid search (vector + keyword)
- Automatic image captioning for retrieval

**Production Patterns:**
- Use CLIP for unified text-image embeddings
- Generate captions for images to enable text search
- Implement hybrid search for better recall
- Cache results aggressively
- Re-rank results with cross-encoders
- Monitor query patterns and performance

**Best Practices:**
- Preprocess and optimize images
- Chunk documents while preserving image associations
- Use appropriate embedding models (CLIP for multi-modal)
- Implement caching to reduce costs
- Monitor retrieval quality
- Re-rank for accuracy

**Applications:**
- Technical documentation with diagrams
- Product catalogs with images
- Research papers with figures
- Medical records with scans
- Educational content with illustrations
- Visual question answering over document collections

Next, we'll explore cross-modal generation, creating content in one modality from another.
`,
  codeExamples: [
    {
      title: 'Multi-Modal RAG with CLIP',
      description:
        'Complete multi-modal RAG system using CLIP embeddings for unified text-image retrieval',
      language: 'python',
      code: `# See MultiModalRAG class in content above`,
    },
  ],
  practicalTips: [
    'Use CLIP embeddings for unified text-image search in the same vector space',
    'Generate detailed captions for images to enable text-based retrieval',
    'Implement hybrid search (vector + keyword) for better recall and precision',
    'Cache query results aggressively - RAG queries are expensive',
    'Re-rank retrieved results with cross-encoders for better accuracy',
    'Preprocess images: resize to 1024px max, convert to RGB, optimize',
    'Chunk documents while preserving image-text associations',
    'Limit to 3-5 images per LLM query to control costs',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/multi-modal-rag',
};
