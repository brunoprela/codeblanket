export const buildingDocumentProcessingSystem = {
  title: 'Building a Document Processing System',
  id: 'building-document-processing-system',
  content: `
# Building a Document Processing System

## Introduction

A production document processing system must handle any file type users throw at it—PDFs, Word docs, Excel spreadsheets, images, even scanned documents with OCR. This section covers building a universal document processor that extracts structured data, understands content, and makes it queryable with AI.

Modern document processing systems power applications like:
- **DocuSign**: Contract analysis and extraction
- **Notion**: Universal file import
- **ChatPDF**: Document Q&A
- **LlamaIndex/LangChain**: Document loaders
- **Enterprise search**: Unified document search

### What We'll Build

A complete document processing pipeline that:
- Accepts any file type (PDF, DOCX, XLSX, images, etc.)
- Extracts text, tables, images with high accuracy
- Performs OCR on scanned documents
- Chunks content intelligently for LLM processing
- Creates embeddings for semantic search
- Provides Q&A interface over documents
- Handles batch processing at scale

### Architecture Overview

\`\`\`
┌──────────────────────────────────────────────────────────────┐
│          Document Processing System                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐       ┌──────────────┐                     │
│  │   Upload    │──────▶│  File Type   │                     │
│  │  Interface  │       │  Detection   │                     │
│  └─────────────┘       └──────┬───────┘                     │
│                               │                              │
│                ┌──────────────┼──────────────┐              │
│                │              │              │              │
│         ┌──────▼─────┐ ┌─────▼──────┐ ┌────▼─────┐        │
│         │   PDF      │ │   DOCX     │ │   XLSX   │        │
│         │  Processor │ │  Processor │ │ Processor│        │
│         └──────┬─────┘ └─────┬──────┘ └────┬─────┘        │
│                │              │              │              │
│                └──────────────┼──────────────┘              │
│                               │                              │
│                        ┌──────▼────────┐                    │
│                        │  OCR Engine   │                    │
│                        │  (Tesseract)  │                    │
│                        └──────┬────────┘                    │
│                               │                              │
│                        ┌──────▼────────┐                    │
│                        │   Content     │                    │
│                        │   Chunker     │                    │
│                        └──────┬────────┘                    │
│                               │                              │
│                  ┌────────────┼────────────┐                │
│                  │                         │                │
│          ┌───────▼────────┐        ┌──────▼──────┐        │
│          │   Embedding    │        │   Storage   │        │
│          │   Generation   │        │ (PostgreSQL)│        │
│          └───────┬────────┘        └──────┬──────┘        │
│                  │                         │                │
│          ┌───────▼────────┐        ┌──────▼──────┐        │
│          │  Vector DB     │◄───────│  Metadata   │        │
│          │  (Pinecone)    │        │   Index     │        │
│          └────────────────┘        └─────────────┘        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │             Q&A Interface                            │  │
│  │  • Semantic search across documents                 │  │
│  │  • Context-aware question answering                 │  │
│  │  • Citation and source tracking                     │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
\`\`\`

---

## File Type Detection

### Universal File Handler

\`\`\`python
"""
File Type Detection and Router
"""

import magic
from pathlib import Path
from typing import Optional, Dict
from enum import Enum

class FileType(Enum):
    """Supported file types"""
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOC = "application/msword"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    XLS = "application/vnd.ms-excel"
    CSV = "text/csv"
    TXT = "text/plain"
    MD = "text/markdown"
    HTML = "text/html"
    IMAGE = "image/*"
    UNKNOWN = "unknown"

class FileDetector:
    """
    Detect file type using magic numbers (not just extension)
    """
    
    def __init__(self):
        self.magic = magic.Magic (mime=True)
    
    def detect (self, file_path: Path) -> FileType:
        """
        Detect file type from content, not extension
        """
        mime_type = self.magic.from_file (str (file_path))
        
        # Map MIME types to FileType enum
        if "pdf" in mime_type:
            return FileType.PDF
        elif "wordprocessingml" in mime_type:
            return FileType.DOCX
        elif "msword" in mime_type:
            return FileType.DOC
        elif "spreadsheetml" in mime_type:
            return FileType.XLSX
        elif "ms-excel" in mime_type:
            return FileType.XLS
        elif mime_type == "text/csv":
            return FileType.CSV
        elif "text/plain" in mime_type or "text/markdown" in mime_type:
            # Check extension for .md files
            if file_path.suffix == ".md":
                return FileType.MD
            return FileType.TXT
        elif "text/html" in mime_type:
            return FileType.HTML
        elif "image/" in mime_type:
            return FileType.IMAGE
        else:
            return FileType.UNKNOWN
    
    def is_processable (self, file_path: Path) -> bool:
        """Check if file can be processed"""
        file_type = self.detect (file_path)
        return file_type != FileType.UNKNOWN

# Usage
detector = FileDetector()
file_type = detector.detect(Path("document.pdf"))
print(f"Detected type: {file_type}")

# Important: Don't trust file extensions!
# A .txt file could actually be a PDF with renamed extension
\`\`\`

---

## PDF Processing

### Advanced PDF Extractor

\`\`\`python
"""
Comprehensive PDF Processing
"""

from pypdf import PdfReader
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import pytesseract

class PDFContent(BaseModel):
    """Extracted PDF content"""
    text: str
    pages: List[Dict[str, any]]
    tables: List[Dict[str, any]]
    images: List[Dict[str, any]]
    metadata: Dict[str, any]
    is_scanned: bool

class PDFProcessor:
    """
    Multi-method PDF extraction for maximum reliability
    """
    
    def __init__(self):
        self.ocr_enabled = True
        
    async def process (self, file_path: Path) -> PDFContent:
        """
        Process PDF with multiple extraction methods
        """
        # Try PyPDF2 first (fastest)
        text, metadata = self._extract_with_pypdf (file_path)
        
        # If little text found, likely scanned - use OCR
        is_scanned = len (text.strip()) < 100
        
        if is_scanned and self.ocr_enabled:
            print(f"Scanned PDF detected, using OCR...")
            text = await self._extract_with_ocr (file_path)
        
        # Extract tables with pdfplumber
        tables = self._extract_tables (file_path)
        
        # Extract images
        images = self._extract_images (file_path)
        
        # Extract per-page content
        pages = self._extract_pages (file_path)
        
        return PDFContent(
            text=text,
            pages=pages,
            tables=tables,
            images=images,
            metadata=metadata,
            is_scanned=is_scanned
        )
    
    def _extract_with_pypdf (self, file_path: Path) -> Tuple[str, Dict]:
        """Extract text with PyPDF2"""
        reader = PdfReader (file_path)
        
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\\n\\n"
        
        metadata = {
            "pages": len (reader.pages),
            "title": reader.metadata.get('/Title', '),
            "author": reader.metadata.get('/Author', '),
            "subject": reader.metadata.get('/Subject', '),
            "creator": reader.metadata.get('/Creator', ')
        }
        
        return text, metadata
    
    async def _extract_with_ocr (self, file_path: Path) -> str:
        """
        Extract text using OCR (for scanned PDFs)
        """
        doc = fitz.open (file_path)
        text = ""
        
        for page_num, page in enumerate (doc):
            # Convert page to image
            pix = page.get_pixmap (dpi=300)  # High DPI for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Apply OCR
            page_text = pytesseract.image_to_string (img, lang='eng')
            text += f"[Page {page_num + 1}]\\n{page_text}\\n\\n"
        
        doc.close()
        return text
    
    def _extract_tables (self, file_path: Path) -> List[Dict]:
        """Extract tables with pdfplumber"""
        tables = []
        
        with pdfplumber.open (file_path) as pdf:
            for page_num, page in enumerate (pdf.pages):
                page_tables = page.extract_tables()
                
                for table_num, table in enumerate (page_tables):
                    if table:
                        tables.append({
                            "page": page_num + 1,
                            "table_number": table_num + 1,
                            "data": table,
                            "headers": table[0] if table else [],
                            "rows": table[1:] if len (table) > 1 else []
                        })
        
        return tables
    
    def _extract_images (self, file_path: Path) -> List[Dict]:
        """Extract images from PDF"""
        doc = fitz.open (file_path)
        images = []
        
        for page_num, page in enumerate (doc):
            image_list = page.get_images()
            
            for img_num, img in enumerate (image_list):
                xref = img[0]
                base_image = doc.extract_image (xref)
                
                images.append({
                    "page": page_num + 1,
                    "image_number": img_num + 1,
                    "format": base_image["ext"],
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "data": base_image["image"]  # Binary data
                })
        
        doc.close()
        return images
    
    def _extract_pages (self, file_path: Path) -> List[Dict]:
        """Extract per-page content"""
        reader = PdfReader (file_path)
        pages = []
        
        for i, page in enumerate (reader.pages):
            pages.append({
                "page_number": i + 1,
                "text": page.extract_text(),
                "word_count": len (page.extract_text().split())
            })
        
        return pages

# Usage
processor = PDFProcessor()
content = await processor.process(Path("document.pdf"))

print(f"Extracted {len (content.text)} characters")
print(f"Found {len (content.tables)} tables")
print(f"Found {len (content.images)} images")
print(f"Scanned: {content.is_scanned}")
\`\`\`

---

## Word Document Processing

### DOCX Processor

\`\`\`python
"""
Word Document (.docx) Processing
"""

from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

class DOCXContent(BaseModel):
    """Extracted DOCX content"""
    text: str
    paragraphs: List[Dict[str, any]]
    tables: List[List[List[str]]]
    images: List[Dict[str, any]]
    styles: Dict[str, int]
    metadata: Dict[str, any]

class DOCXProcessor:
    """
    Process Word documents preserving structure
    """
    
    def process (self, file_path: Path) -> DOCXContent:
        """
        Extract structured content from DOCX
        """
        doc = Document (file_path)
        
        # Extract text maintaining structure
        paragraphs = self._extract_paragraphs (doc)
        
        # Extract tables
        tables = self._extract_tables (doc)
        
        # Extract images
        images = self._extract_images (doc)
        
        # Analyze styles used
        styles = self._analyze_styles (doc)
        
        # Get metadata
        metadata = self._extract_metadata (doc)
        
        # Combine all text
        full_text = "\\n\\n".join([p["text"] for p in paragraphs])
        
        return DOCXContent(
            text=full_text,
            paragraphs=paragraphs,
            tables=tables,
            images=images,
            styles=styles,
            metadata=metadata
        )
    
    def _extract_paragraphs (self, doc: Document) -> List[Dict]:
        """Extract paragraphs with formatting"""
        paragraphs = []
        
        for i, para in enumerate (doc.paragraphs):
            if para.text.strip():
                paragraphs.append({
                    "index": i,
                    "text": para.text,
                    "style": para.style.name if para.style else "Normal",
                    "alignment": str (para.alignment) if para.alignment else "LEFT",
                    "is_heading": para.style.name.startswith("Heading") if para.style else False,
                    "level": self._get_heading_level (para)
                })
        
        return paragraphs
    
    def _get_heading_level (self, para: Paragraph) -> Optional[int]:
        """Get heading level (1-9)"""
        if para.style and para.style.name.startswith("Heading"):
            try:
                return int (para.style.name.replace("Heading ", ""))
            except:
                return None
        return None
    
    def _extract_tables (self, doc: Document) -> List[List[List[str]]]:
        """Extract all tables"""
        tables = []
        
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append (cell.text.strip())
                table_data.append (row_data)
            tables.append (table_data)
        
        return tables
    
    def _extract_images (self, doc: Document) -> List[Dict]:
        """Extract embedded images"""
        images = []
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                images.append({
                    "filename": rel.target_ref.split("/")[-1],
                    "content_type": rel.target_part.content_type
                })
        
        return images
    
    def _analyze_styles (self, doc: Document) -> Dict[str, int]:
        """Count usage of different styles"""
        styles = {}
        
        for para in doc.paragraphs:
            if para.style:
                style_name = para.style.name
                styles[style_name] = styles.get (style_name, 0) + 1
        
        return styles
    
    def _extract_metadata (self, doc: Document) -> Dict[str, any]:
        """Extract document metadata"""
        core_props = doc.core_properties
        
        return {
            "title": core_props.title or "",
            "author": core_props.author or "",
            "subject": core_props.subject or "",
            "keywords": core_props.keywords or "",
            "created": core_props.created.isoformat() if core_props.created else None,
            "modified": core_props.modified.isoformat() if core_props.modified else None,
            "revision": core_props.revision,
            "last_modified_by": core_props.last_modified_by or ""
        }

# Usage
docx_processor = DOCXProcessor()
content = docx_processor.process(Path("document.docx"))

print(f"Paragraphs: {len (content.paragraphs)}")
print(f"Tables: {len (content.tables)}")
print(f"Images: {len (content.images)}")
print(f"Styles used: {content.styles}")
\`\`\`

---

## Excel Processing

### Comprehensive Excel Processor

\`\`\`python
"""
Excel Spreadsheet Processing
"""

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

class ExcelContent(BaseModel):
    """Extracted Excel content"""
    sheets: List[Dict[str, any]]
    summary: str
    total_rows: int
    total_cols: int
    formulas: List[str]

class ExcelProcessor:
    """
    Process Excel files with formula and formatting preservation
    """
    
    def process (self, file_path: Path) -> ExcelContent:
        """
        Extract structured data from Excel
        """
        # Load with pandas for data
        excel_file = pd.ExcelFile (file_path)
        
        # Load with openpyxl for formulas
        workbook = load_workbook (file_path, data_only=False)
        
        sheets = []
        total_rows = 0
        total_cols = 0
        all_formulas = []
        
        for sheet_name in excel_file.sheet_names:
            # Get data
            df = pd.read_excel (excel_file, sheet_name=sheet_name)
            
            # Get formulas
            ws = workbook[sheet_name]
            formulas = self._extract_formulas (ws)
            all_formulas.extend (formulas)
            
            # Get cell formatting
            formatting = self._extract_formatting (ws)
            
            sheets.append({
                "name": sheet_name,
                "data": df.to_dict (orient='records'),
                "shape": df.shape,
                "columns": list (df.columns),
                "formulas": formulas,
                "formatting": formatting,
                "has_charts": len (ws._charts) > 0
            })
            
            total_rows += df.shape[0]
            total_cols += df.shape[1]
        
        # Generate summary
        summary = self._generate_summary (sheets)
        
        return ExcelContent(
            sheets=sheets,
            summary=summary,
            total_rows=total_rows,
            total_cols=total_cols,
            formulas=all_formulas
        )
    
    def _extract_formulas (self, worksheet) -> List[str]:
        """Extract all formulas from worksheet"""
        formulas = []
        
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.value and isinstance (cell.value, str) and cell.value.startswith('='):
                    formulas.append({
                        "cell": cell.coordinate,
                        "formula": cell.value
                    })
        
        return formulas
    
    def _extract_formatting (self, worksheet) -> Dict:
        """Extract cell formatting information"""
        formatting = {
            "bold_cells": [],
            "colored_cells": [],
            "merged_cells": []
        }
        
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.font and cell.font.bold:
                    formatting["bold_cells"].append (cell.coordinate)
                
                if cell.fill and cell.fill.start_color:
                    formatting["colored_cells"].append({
                        "cell": cell.coordinate,
                        "color": cell.fill.start_color.rgb
                    })
        
        # Get merged cells
        formatting["merged_cells"] = [str (r) for r in worksheet.merged_cells.ranges]
        
        return formatting
    
    def _generate_summary (self, sheets: List[Dict]) -> str:
        """Generate natural language summary"""
        summary_parts = []
        
        summary_parts.append (f"Excel file with {len (sheets)} sheet (s):")
        
        for sheet in sheets:
            name = sheet["name"]
            rows, cols = sheet["shape"]
            formulas_count = len (sheet["formulas"])
            
            summary_parts.append(
                f"- {name}: {rows} rows × {cols} columns"
            )
            
            if formulas_count > 0:
                summary_parts.append (f"  Contains {formulas_count} formulas")
            
            if sheet["has_charts"]:
                summary_parts.append (f"  Includes charts/graphs")
        
        return "\\n".join (summary_parts)

# Usage
excel_processor = ExcelProcessor()
content = excel_processor.process(Path("spreadsheet.xlsx"))

print(content.summary)
print(f"Total formulas: {len (content.formulas)}")
\`\`\`

---

## Intelligent Chunking

### Context-Aware Chunking Strategy

\`\`\`python
"""
Intelligent document chunking for LLM processing
"""

import tiktoken
from typing import List

class DocumentChunk(BaseModel):
    """Document chunk with metadata"""
    content: str
    chunk_id: str
    start_char: int
    end_char: int
    tokens: int
    metadata: Dict[str, any]
    overlap_with_previous: int = 0

class SmartChunker:
    """
    Context-aware chunking that preserves semantic boundaries
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model: str = "gpt-4"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.encoding_for_model (model)
    
    def chunk_document(
        self,
        text: str,
        metadata: Dict = None
    ) -> List[DocumentChunk]:
        """
        Chunk document intelligently
        """
        # Split into paragraphs first
        paragraphs = self._split_paragraphs (text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        char_position = 0
        
        for para in paragraphs:
            para_tokens = len (self.encoder.encode (para))
            
            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append (self._create_chunk(
                        "\\n\\n".join (current_chunk),
                        len (chunks),
                        char_position,
                        metadata
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = self._split_sentences (para)
                for sentence in sentences:
                    sent_tokens = len (self.encoder.encode (sentence))
                    
                    if current_tokens + sent_tokens > self.chunk_size:
                        # Create chunk
                        chunks.append (self._create_chunk(
                            "\\n\\n".join (current_chunk),
                            len (chunks),
                            char_position,
                            metadata
                        ))
                        
                        # Keep overlap from previous chunk
                        overlap_text = self._get_overlap (current_chunk)
                        current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                        current_tokens = len (self.encoder.encode(" ".join (current_chunk)))
                    else:
                        current_chunk.append (sentence)
                        current_tokens += sent_tokens
            
            # Normal paragraph that fits
            elif current_tokens + para_tokens <= self.chunk_size:
                current_chunk.append (para)
                current_tokens += para_tokens
            
            # Paragraph would exceed limit, start new chunk
            else:
                chunks.append (self._create_chunk(
                    "\\n\\n".join (current_chunk),
                    len (chunks),
                    char_position,
                    metadata
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap (current_chunk)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = len (self.encoder.encode(" ".join (current_chunk)))
            
            char_position += len (para) + 2  # +2 for \\n\\n
        
        # Add final chunk
        if current_chunk:
            chunks.append (self._create_chunk(
                "\\n\\n".join (current_chunk),
                len (chunks),
                char_position,
                metadata
            ))
        
        return chunks
    
    def _split_paragraphs (self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or markdown headers
        paragraphs = []
        current = []
        
        for line in text.split('\\n'):
            if line.strip() == ' and current:
                paragraphs.append('\\n'.join (current))
                current = []
            elif line.strip():
                current.append (line)
        
        if current:
            paragraphs.append('\\n'.join (current))
        
        return paragraphs
    
    def _split_sentences (self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitter
        sentences = re.split (r'(?<=[.!?])\\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap (self, current_chunk: List[str]) -> str:
        """Get overlap text from previous chunk"""
        overlap_text = " ".join (current_chunk)
        overlap_tokens = self.encoder.encode (overlap_text)
        
        if len (overlap_tokens) > self.chunk_overlap:
            # Take last N tokens
            overlap_tokens = overlap_tokens[-self.chunk_overlap:]
            return self.encoder.decode (overlap_tokens)
        
        return overlap_text
    
    def _create_chunk(
        self,
        content: str,
        chunk_id: int,
        start_char: int,
        metadata: Dict
    ) -> DocumentChunk:
        """Create chunk object"""
        tokens = len (self.encoder.encode (content))
        
        return DocumentChunk(
            content=content,
            chunk_id=f"chunk_{chunk_id}",
            start_char=start_char,
            end_char=start_char + len (content),
            tokens=tokens,
            metadata=metadata or {}
        )

# Usage
chunker = SmartChunker (chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk_document (long_document_text, metadata={"source": "paper.pdf"})

print(f"Created {len (chunks)} chunks")
for chunk in chunks[:3]:
    print(f"Chunk {chunk.chunk_id}: {chunk.tokens} tokens")
\`\`\`

---

## Embedding Generation

### Create Searchable Embeddings

\`\`\`python
"""
Generate embeddings for semantic search
"""

from openai import AsyncOpenAI
import numpy as np
from typing import List
import asyncio

class EmbeddingGenerator:
    """
    Generate embeddings for document chunks
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.client = llm_client
        self.model = "text-embedding-3-small"  # Cheaper, good quality
        self.batch_size = 100  # OpenAI allows batching
    
    async def generate_embeddings(
        self,
        chunks: List[DocumentChunk]
    ) -> List[Dict[str, any]]:
        """
        Generate embeddings for all chunks
        """
        embeddings = []
        
        # Process in batches for efficiency
        for i in range(0, len (chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch (batch)
            embeddings.extend (batch_embeddings)
        
        return embeddings
    
    async def _embed_batch(
        self,
        chunks: List[DocumentChunk]
    ) -> List[Dict[str, any]]:
        """
        Generate embeddings for a batch of chunks
        """
        texts = [chunk.content for chunk in chunks]
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        embeddings = []
        for i, embedding_data in enumerate (response.data):
            embeddings.append({
                "chunk_id": chunks[i].chunk_id,
                "embedding": embedding_data.embedding,
                "dimension": len (embedding_data.embedding),
                "metadata": chunks[i].metadata
            })
        
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between embeddings
        """
        vec1 = np.array (embedding1)
        vec2 = np.array (embedding2)
        
        return np.dot (vec1, vec2) / (np.linalg.norm (vec1) * np.linalg.norm (vec2))

# Usage
embedding_gen = EmbeddingGenerator (llm_client)
embeddings = await embedding_gen.generate_embeddings (chunks)

print(f"Generated {len (embeddings)} embeddings")
print(f"Dimension: {embeddings[0]['dimension']}")
\`\`\`

---

## Storage & Indexing

### Vector Database Integration

\`\`\`python
"""
Store chunks and embeddings in Pinecone
"""

from pinecone import Pinecone, ServerlessSpec
import hashlib

class DocumentStore:
    """
    Store documents with vector search capability
    """
    
    def __init__(self, pinecone_api_key: str, index_name: str = "documents"):
        self.pc = Pinecone (api_key=pinecone_api_key)
        self.index_name = index_name
        self.index = None
        
        self._init_index()
    
    def _init_index (self):
        """Initialize or connect to Pinecone index"""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # text-embedding-3-small dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.index = self.pc.Index (self.index_name)
    
    async def store_document(
        self,
        document_id: str,
        chunks: List[DocumentChunk],
        embeddings: List[Dict[str, any]]
    ):
        """
        Store document chunks with embeddings
        """
        vectors = []
        
        for chunk, emb in zip (chunks, embeddings):
            vector_id = f"{document_id}_{chunk.chunk_id}"
            
            vectors.append({
                "id": vector_id,
                "values": emb["embedding"],
                "metadata": {
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content[:1000],  # Store excerpt
                    "tokens": chunk.tokens,
                    "start_char": chunk.start_char,
                    **chunk.metadata
                }
            })
        
        # Batch upsert
        self.index.upsert (vectors=vectors, batch_size=100)
        
        print(f"Stored {len (vectors)} chunks for document {document_id}")
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Dict = None
    ) -> List[Dict]:
        """
        Search for similar chunks
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            }
            for match in results["matches"]
        ]
    
    async def delete_document (self, document_id: str):
        """Delete all chunks for a document"""
        self.index.delete (filter={"document_id": document_id})

# Usage
store = DocumentStore (pinecone_api_key)
await store.store_document("doc_123", chunks, embeddings)

# Search
query_emb = await embedding_gen.generate_embeddings([DocumentChunk (content="quantum computing")])
results = await store.search (query_emb[0]["embedding"], top_k=5)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['metadata']['content'][:200]}...")
\`\`\`

---

## Q&A Interface

### Document Question Answering

\`\`\`python
"""
Q&A system over processed documents
"""

class DocumentQA:
    """
    Question answering system over documents
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        embedding_gen: EmbeddingGenerator,
        doc_store: DocumentStore
    ):
        self.llm = llm_client
        self.embedding_gen = embedding_gen
        self.store = doc_store
    
    async def answer_question(
        self,
        question: str,
        document_ids: List[str] = None,
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Answer question using relevant document chunks
        """
        # 1. Generate question embedding
        question_embedding = await self.embedding_gen._embed_batch([
            DocumentChunk(
                content=question,
                chunk_id="query",
                start_char=0,
                end_char=len (question),
                tokens=0
            )
        ])
        
        # 2. Search for relevant chunks
        filter_dict = {"document_id": {"$in": document_ids}} if document_ids else None
        results = await self.store.search(
            question_embedding[0]["embedding"],
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # 3. Build context from top results
        context = self._build_context (results)
        
        # 4. Generate answer with citations
        answer = await self._generate_answer (question, context, results)
        
        return answer
    
    def _build_context (self, results: List[Dict]) -> str:
        """Build context string from search results"""
        context_parts = []
        
        for i, result in enumerate (results):
            context_parts.append(
                f"[Source {i+1}]\\n{result['metadata']['content']}"
            )
        
        return "\\n\\n".join (context_parts)
    
    async def _generate_answer(
        self,
        question: str,
        context: str,
        results: List[Dict]
    ) -> Dict[str, any]:
        """Generate answer with LLM"""
        prompt = f"""Answer the question based on the provided context. Include citations to sources.

Context:
{context}

Question: {question}

Provide:
1. Direct answer
2. Supporting evidence with source citations [Source N]
3. Confidence level (0.0 to 1.0)

Answer:"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        answer_text = response.choices[0].message.content
        
        return {
            "answer": answer_text,
            "sources": [
                {
                    "content": r["metadata"]["content"],
                    "document_id": r["metadata"]["document_id"],
                    "score": r["score"]
                }
                for r in results
            ],
            "context_used": context
        }

# Usage
qa = DocumentQA(llm_client, embedding_gen, doc_store)

answer = await qa.answer_question(
    "What are the main applications of quantum computing?",
    document_ids=["doc_123", "doc_456"]
)

print(f"Answer: {answer['answer']}")
print(f"Sources used: {len (answer['sources'])}")
\`\`\`

---

## Complete Pipeline

### End-to-End Document Processing

\`\`\`python
"""
Complete document processing pipeline
"""

class DocumentProcessingPipeline:
    """
    Orchestrates entire document processing workflow
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        pinecone_api_key: str,
        serper_api_key: str = None
    ):
        self.file_detector = FileDetector()
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.excel_processor = ExcelProcessor()
        self.chunker = SmartChunker()
        self.embedding_gen = EmbeddingGenerator (llm_client)
        self.doc_store = DocumentStore (pinecone_api_key)
        self.qa = DocumentQA(llm_client, self.embedding_gen, self.doc_store)
    
    async def process_document(
        self,
        file_path: Path,
        document_id: str = None
    ) -> str:
        """
        Process document end-to-end
        """
        if not document_id:
            document_id = hashlib.sha256(file_path.read_bytes()).hexdigest()
        
        print(f"Processing {file_path.name}...")
        
        # 1. Detect file type
        file_type = self.file_detector.detect (file_path)
        print(f"Detected type: {file_type}")
        
        # 2. Extract content
        if file_type == FileType.PDF:
            content = await self.pdf_processor.process (file_path)
            text = content.text
            metadata = content.metadata
        elif file_type == FileType.DOCX:
            content = self.docx_processor.process (file_path)
            text = content.text
            metadata = content.metadata
        elif file_type == FileType.XLSX:
            content = self.excel_processor.process (file_path)
            text = content.summary
            metadata = {"sheets": len (content.sheets)}
        else:
            text = file_path.read_text()
            metadata = {}
        
        print(f"Extracted {len (text)} characters")
        
        # 3. Chunk content
        metadata.update({"filename": file_path.name, "type": file_type.value})
        chunks = self.chunker.chunk_document (text, metadata)
        print(f"Created {len (chunks)} chunks")
        
        # 4. Generate embeddings
        embeddings = await self.embedding_gen.generate_embeddings (chunks)
        print(f"Generated {len (embeddings)} embeddings")
        
        # 5. Store in vector database
        await self.doc_store.store_document (document_id, chunks, embeddings)
        print(f"Stored document {document_id}")
        
        return document_id
    
    async def process_batch(
        self,
        file_paths: List[Path]
    ) -> List[str]:
        """
        Process multiple documents in parallel
        """
        tasks = [
            self.process_document (path)
            for path in file_paths
        ]
        
        document_ids = await asyncio.gather(*tasks)
        return document_ids

# Complete usage example
async def main():
    pipeline = DocumentProcessingPipeline(
        llm_client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # Process single document
    doc_id = await pipeline.process_document(Path("research_paper.pdf"))
    
    # Ask questions
    answer = await pipeline.qa.answer_question(
        "What are the main findings of this research?",
        document_ids=[doc_id]
    )
    
    print(f"Answer: {answer['answer']}")

if __name__ == "__main__":
    asyncio.run (main())
\`\`\`

---

## Conclusion

A production document processing system requires:

1. **Universal File Support**: Handle any file type users provide
2. **Robust Extraction**: Multiple methods (PyPDF, OCR, etc.) for reliability
3. **Smart Chunking**: Preserve semantic boundaries, not arbitrary splits
4. **Vector Search**: Enable semantic search across documents
5. **Q&A Interface**: Let users query documents naturally
6. **Batch Processing**: Handle large volumes efficiently
7. **Error Handling**: Gracefully handle corrupted files

**Key Technologies**:
- **PyPDF2/pdfplumber**: PDF extraction
- **python-docx**: Word documents
- **openpyxl/pandas**: Excel files
- **Tesseract**: OCR for scanned documents
- **OpenAI embeddings**: Semantic search
- **Pinecone**: Vector database

This system powers applications like ChatPDF, document search engines, and knowledge management platforms.
`,
};
