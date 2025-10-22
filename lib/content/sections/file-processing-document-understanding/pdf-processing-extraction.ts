/**
 * PDF Processing & Extraction Section  
 * Module 3: File Processing & Document Understanding
 */

export const pdfprocessingextractionSection = {
    id: 'pdf-processing-extraction',
    title: 'PDF Processing & Extraction',
    content: `# PDF Processing & Extraction

Master PDF file processing for building AI applications that extract and understand document content from PDFs.

## Overview: PDFs in LLM Applications

PDFs are everywhere - from research papers to invoices to legal documents. Building LLM applications that can intelligently process PDFs unlocks enormous value. However, PDFs are notoriously difficult to parse because they're designed for visual presentation, not data extraction.

**Challenges:**
- PDFs store visual positioning, not semantic structure
- Text may be images (scanned PDFs)
- Tables are hard to extract correctly
- Mixed content: text, images, tables, forms
- Encrypted/password-protected PDFs
- Multi-column layouts

**Use Cases:**
- Document Q&A systems
- Invoice processing
- Research paper analysis
- Legal document review
- Form data extraction

## PDF Libraries Overview

\`\`\`python
# PyPDF2: Basic PDF operations
# - Lightweight, pure Python
# - Good for simple text extraction
# - Can merge/split PDFs
# - Struggles with complex layouts

# pdfplumber: Advanced extraction
# - Excellent table extraction
# - Layout-aware text extraction
# - Visual debugging capabilities
# - Best for structured documents

# pdfminer.six: Low-level PDF parsing
# - Most accurate text positioning
# - Extract font and layout info
# - Complex API
# - Best for custom extraction logic

# PyMuPDF (fitz): Fast and feature-rich
# - Fastest PDF library
# - Image extraction
# - Rendering capabilities
# - Best for performance-critical apps
\`\`\`

## Basic Text Extraction

### Using PyPDF2

\`\`\`python
# pip install PyPDF2
from PyPDF2 import PdfReader
from pathlib import Path

def extract_text_pypdf2(filepath: str) -> str:
    """Extract text from PDF using PyPDF2."""
    reader = PdfReader(filepath)
    
    text_content = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        text_content.append(f"--- Page {page_num + 1} ---\\n{text}")
    
    return "\\n\\n".join(text_content)

# Usage
text = extract_text_pypdf2("document.pdf")
print(f"Extracted {len(text)} characters")
\`\`\`

### Using pdfplumber

\`\`\`python
# pip install pdfplumber
import pdfplumber
from typing import List, Dict

def extract_text_pdfplumber(filepath: str) -> str:
    """Extract text with better layout preservation."""
    text_content = []
    
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_content.append(f"--- Page {page_num + 1} ---\\n{text}")
    
    return "\\n\\n".join(text_content)

def extract_text_with_layout(filepath: str) -> List[Dict]:
    """Extract text with position information."""
    pages_data = []
    
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            pages_data.append({
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "words": words,
                "text": page.extract_text()
            })
    
    return pages_data

# Usage
text = extract_text_pdfplumber("document.pdf")
layout_data = extract_text_with_layout("document.pdf")
\`\`\`

## Table Extraction

### Extracting Tables with pdfplumber

\`\`\`python
import pdfplumber
import pandas as pd
from typing import List

def extract_tables(filepath: str) -> List[pd.DataFrame]:
    """
    Extract all tables from PDF.
    
    Returns list of DataFrames, one per table found.
    """
    all_tables = []
    
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if table:
                    # Convert to DataFrame
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df['source_page'] = page_num + 1
                    df['table_index'] = table_idx
                    all_tables.append(df)
    
    return all_tables

def extract_tables_with_settings(filepath: str) -> List[pd.DataFrame]:
    """Extract tables with custom settings for better accuracy."""
    all_tables = []
    
    table_settings = {
        "vertical_strategy": "lines",  # or "text"
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
    }
    
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables(table_settings=table_settings)
            
            for table in tables:
                if table and len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    all_tables.append(df)
    
    return all_tables

# Usage
tables = extract_tables("financial_report.pdf")
for i, df in enumerate(tables):
    print(f"\\nTable {i + 1}:")
    print(df.head())
\`\`\`

### Advanced Table Extraction

\`\`\`python
import pdfplumber
import pandas as pd
from typing import List, Dict, Optional

class PDFTableExtractor:
    """
    Production-grade PDF table extractor.
    
    Handles complex table layouts and data cleaning.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.pdf = None
    
    def __enter__(self):
        self.pdf = pdfplumber.open(self.filepath)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pdf:
            self.pdf.close()
    
    def extract_all_tables(self) -> List[Dict]:
        """Extract all tables with metadata."""
        tables_data = []
        
        for page in self.pdf.pages:
            tables = self._extract_tables_from_page(page)
            tables_data.extend(tables)
        
        return tables_data
    
    def _extract_tables_from_page(self, page) -> List[Dict]:
        """Extract tables from a single page."""
        tables_data = []
        tables = page.extract_tables()
        
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            
            # Clean and convert to DataFrame
            df = self._table_to_dataframe(table)
            
            if df is not None:
                tables_data.append({
                    "page_number": page.page_number,
                    "table_index": table_idx,
                    "dataframe": df,
                    "row_count": len(df),
                    "column_count": len(df.columns)
                })
        
        return tables_data
    
    def _table_to_dataframe(self, table: List[List]) -> Optional[pd.DataFrame]:
        """Convert table array to cleaned DataFrame."""
        if not table or len(table) < 2:
            return None
        
        # Extract header and data
        header = [str(cell).strip() if cell else f"Column_{i}" 
                  for i, cell in enumerate(table[0])]
        
        # Clean data rows
        data_rows = []
        for row in table[1:]:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            # Skip empty rows
            if any(cleaned_row):
                data_rows.append(cleaned_row)
        
        if not data_rows:
            return None
        
        df = pd.DataFrame(data_rows, columns=header)
        
        # Clean DataFrame
        df = self._clean_dataframe(df)
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame."""
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Strip whitespace
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Try to convert numeric columns
        for col in df.columns:
            try:
                # Remove currency symbols and commas
                cleaned = df[col].str.replace('[$,]', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='ignore')
            except:
                pass
        
        return df

# Usage
with PDFTableExtractor("report.pdf") as extractor:
    tables = extractor.extract_all_tables()
    
    for table_data in tables:
        print(f"Page {table_data['page_number']}, Table {table_data['table_index']}")
        print(table_data['dataframe'].head())
\`\`\`

## Handling Scanned PDFs with OCR

### Using Tesseract OCR

\`\`\`python
# pip install pytesseract Pillow pdf2image
# Also need to install tesseract: brew install tesseract (Mac) or apt-get install tesseract-ocr (Linux)

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path

def extract_text_with_ocr(filepath: str) -> str:
    """
    Extract text from scanned PDF using OCR.
    
    For PDFs that are images, not searchable text.
    """
    # Convert PDF pages to images
    images = convert_from_path(filepath)
    
    text_content = []
    
    for page_num, image in enumerate(images, start=1):
        # Run OCR on image
        text = pytesseract.image_to_string(image)
        text_content.append(f"--- Page {page_num} ---\\n{text}")
    
    return "\\n\\n".join(text_content)

def is_pdf_searchable(filepath: str) -> bool:
    """Check if PDF contains searchable text."""
    try:
        text = extract_text_pypdf2(filepath)
        # If we get substantial text, it's searchable
        return len(text.strip()) > 100
    except:
        return False

def extract_text_smart(filepath: str) -> str:
    """
    Smart extraction: use OCR only if needed.
    
    Try text extraction first, fall back to OCR.
    """
    if is_pdf_searchable(filepath):
        print("PDF is searchable, using direct text extraction")
        return extract_text_pdfplumber(filepath)
    else:
        print("PDF appears to be scanned, using OCR")
        return extract_text_with_ocr(filepath)

# Usage
text = extract_text_smart("document.pdf")
\`\`\`

## Extracting Images from PDFs

### Using PyMuPDF

\`\`\`python
# pip install PyMuPDF
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

def extract_images_from_pdf(filepath: str, output_dir: str = "extracted_images") -> List[Dict]:
    """
    Extract all images from PDF.
    
    Returns list of image metadata and saves images to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    doc = fitz.open(filepath)
    images_data = []
    
    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]  # Image reference number
            base_image = doc.extract_image(xref)
            
            if base_image:
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"page_{page_num}_img_{img_idx}.{image_ext}"
                image_path = output_path / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                images_data.append({
                    "page": page_num,
                    "index": img_idx,
                    "filename": image_filename,
                    "path": str(image_path),
                    "format": image_ext,
                    "size": len(image_bytes)
                })
    
    doc.close()
    return images_data

# Usage
images = extract_images_from_pdf("document.pdf")
print(f"Extracted {len(images)} images")
\`\`\`

## PDF Metadata Extraction

\`\`\`python
from PyPDF2 import PdfReader
import pdfplumber
from typing import Dict

def extract_pdf_metadata(filepath: str) -> Dict:
    """Extract comprehensive PDF metadata."""
    reader = PdfReader(filepath)
    metadata = reader.metadata
    
    # Basic metadata
    info = {
        "filename": Path(filepath).name,
        "num_pages": len(reader.pages),
        "author": metadata.get('/Author', ''),
        "creator": metadata.get('/Creator', ''),
        "producer": metadata.get('/Producer', ''),
        "subject": metadata.get('/Subject', ''),
        "title": metadata.get('/Title', ''),
        "creation_date": metadata.get('/CreationDate', ''),
        "modification_date": metadata.get('/ModDate', ''),
    }
    
    # Additional info with pdfplumber
    with pdfplumber.open(filepath) as pdf:
        if pdf.pages:
            first_page = pdf.pages[0]
            info.update({
                "page_width": first_page.width,
                "page_height": first_page.height,
                "has_text": bool(first_page.extract_text())
            })
    
    return info

# Usage
metadata = extract_pdf_metadata("document.pdf")
print(f"Title: {metadata['title']}")
print(f"Pages: {metadata['num_pages']}")
\`\`\`

## Production PDF Processor

\`\`\`python
from pathlib import Path
from typing import Dict, List, Optional
import pdfplumber
import pandas as pd
import logging

class PDFProcessor:
    """
    Production-grade PDF processor for LLM applications.
    
    Extracts text, tables, and metadata from PDFs.
    Foundation for building PDF analysis tools.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_pdf(self, filepath: str) -> Dict:
        """
        Process PDF and extract all content.
        
        Returns comprehensive dictionary with all extracted data.
        """
        path = Path(filepath)
        
        if not path.exists():
            self.logger.error(f"File not found: {filepath}")
            return {}
        
        try:
            result = {
                "filename": path.name,
                "filepath": str(path),
                "metadata": {},
                "pages": [],
                "tables": [],
                "full_text": "",
                "page_count": 0
            }
            
            with pdfplumber.open(filepath) as pdf:
                result["page_count"] = len(pdf.pages)
                result["metadata"] = pdf.metadata or {}
                
                all_text = []
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_data = self._process_page(page, page_num)
                    result["pages"].append(page_data)
                    
                    if page_data["text"]:
                        all_text.append(page_data["text"])
                    
                    if page_data["tables"]:
                        result["tables"].extend(page_data["tables"])
                
                result["full_text"] = "\\n\\n".join(all_text)
            
            self.logger.info(f"Processed {filepath}: {result['page_count']} pages, "
                           f"{len(result['tables'])} tables")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process PDF: {e}")
            return {}
    
    def _process_page(self, page, page_num: int) -> Dict:
        """Process a single page."""
        page_data = {
            "page_number": page_num,
            "text": page.extract_text() or "",
            "tables": [],
            "dimensions": {
                "width": page.width,
                "height": page.height
            }
        }
        
        # Extract tables
        tables = page.extract_tables()
        for table_idx, table in enumerate(tables):
            if table and len(table) > 1:
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    page_data["tables"].append({
                        "page": page_num,
                        "index": table_idx,
                        "dataframe": df,
                        "rows": len(df),
                        "columns": len(df.columns)
                    })
                except:
                    pass
        
        return page_data
    
    def get_summary_for_llm(self, pdf_data: Dict) -> str:
        """
        Generate summary of PDF for LLM context.
        
        Useful for providing PDF structure to LLM before queries.
        """
        if not pdf_data:
            return ""
        
        summary = f"PDF Document: {pdf_data['filename']}\\n"
        summary += f"Pages: {pdf_data['page_count']}\\n"
        summary += f"Tables: {len(pdf_data['tables'])}\\n\\n"
        
        if pdf_data['metadata'].get('title'):
            summary += f"Title: {pdf_data['metadata']['title']}\\n"
        
        # Include first page text as sample
        if pdf_data['pages']:
            first_page_text = pdf_data['pages'][0]['text']
            preview = first_page_text[:500] + "..." if len(first_page_text) > 500 else first_page_text
            summary += f"\\nFirst Page Preview:\\n{preview}\\n"
        
        return summary

# Usage Example: Process PDF for LLM Q&A
processor = PDFProcessor()

# Extract all content
pdf_data = processor.process_pdf("research_paper.pdf")

# Get summary for LLM context
summary = processor.get_summary_for_llm(pdf_data)
print("PDF Summary for LLM:")
print(summary)

# Extract specific data
if pdf_data['tables']:
    print(f"\\nFound {len(pdf_data['tables'])} tables")
    print(pdf_data['tables'][0]['dataframe'].head())

# Get full text for analysis
full_text = pdf_data['full_text']
# Now send to LLM: "Summarize this document: {full_text}"
\`\`\`

## Key Takeaways

1. **Choose the right library** - pdfplumber for tables, PyMuPDF for speed
2. **Handle scanned PDFs** with OCR (Tesseract)
3. **Extract tables carefully** with custom settings for accuracy
4. **Clean extracted data** - remove whitespace, convert types
5. **Test with various PDFs** - layouts differ significantly
6. **Extract images** when visual content matters
7. **Provide PDF structure** to LLMs for better context
8. **Handle errors gracefully** - PDFs can be malformed
9. **Consider file size** - large PDFs need streaming
10. **Validate extracted data** before using in production

These patterns enable building robust PDF processing pipelines for LLM applications, from document Q&A to automated data extraction.`,
    videoUrl: null,
};

