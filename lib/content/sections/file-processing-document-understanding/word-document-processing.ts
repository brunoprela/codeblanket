/**
 * Word Document Processing Section
 * Module 3: File Processing & Document Understanding
 */

export const worddocumentprocessingSection = {
  id: 'word-document-processing',
  title: 'Word Document Processing',
  content: `# Word Document Processing

Master Word document manipulation for building AI applications that can read, modify, and generate professional Word documents.

## Overview: Word Documents in LLM Applications

Word documents (.docx) are the standard for business documents, reports, and contracts. Building LLM applications that can process and generate Word documents enables powerful automation - from contract generation to report writing to document analysis.

**Use Cases:**
- Automated report generation
- Contract template filling
- Document translation and reformatting
- Content extraction for analysis
- Template-based document creation

## Understanding .docx Format

\`\`\`python
# .docx files are ZIP archives containing XML files
# Structure:
# - document.xml: Main document content
# - styles.xml: Document styles
# - media/: Embedded images
# - _rels/: Relationships between parts

# python-docx library handles this complexity for us
\`\`\`

## Reading Word Documents

### Basic Text Extraction

\`\`\`python
# pip install python-docx
from docx import Document
from pathlib import Path

def read_word_document(filepath: str) -> str:
    """Extract all text from Word document."""
    doc = Document(filepath)
    
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    
    return "\\n\\n".join(paragraphs)

# Usage
text = read_word_document("report.docx")
print(text)
\`\`\`

### Reading with Structure Preservation

\`\`\`python
from docx import Document
from typing import List, Dict

def read_document_structured(filepath: str) -> Dict:
    """
    Read document preserving structure.
    
    Returns paragraphs with their formatting info.
    """
    doc = Document(filepath)
    
    content = {
        "paragraphs": [],
        "tables": [],
        "images": []
    }
    
    # Extract paragraphs with formatting
    for para in doc.paragraphs:
        para_info = {
            "text": para.text,
            "style": para.style.name,
            "bold": any(run.bold for run in para.runs),
            "italic": any(run.italic for run in para.runs),
            "alignment": str(para.alignment) if para.alignment else None
        }
        content["paragraphs"].append(para_info)
    
    # Extract tables
    for table_idx, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        
        content["tables"].append({
            "index": table_idx,
            "data": table_data,
            "rows": len(table.rows),
            "cols": len(table.columns)
        })
    
    return content

# Usage
structured_content = read_document_structured("document.docx")
print(f"Found {len(structured_content['paragraphs'])} paragraphs")
print(f"Found {len(structured_content['tables'])} tables")
\`\`\`

## Extracting Tables from Word

\`\`\`python
from docx import Document
import pandas as pd
from typing import List

def extract_tables_from_word(filepath: str) -> List[pd.DataFrame]:
    """Extract all tables as DataFrames."""
    doc = Document(filepath)
    tables = []
    
    for table in doc.tables:
        data = []
        
        # Extract all rows
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            data.append(row_data)
        
        if len(data) > 1:
            # First row as headers
            df = pd.DataFrame(data[1:], columns=data[0])
            tables.append(df)
    
    return tables

# Usage
tables = extract_tables_from_word("financial_report.docx")
for i, df in enumerate(tables):
    print(f"\\nTable {i+1}:")
    print(df.head())
\`\`\`

## Creating Word Documents

### Basic Document Creation

\`\`\`python
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_simple_document(output_path: str):
    """Create a simple Word document."""
    doc = Document()
    
    # Add title
    title = doc.add_heading('Document Title', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add paragraph
    doc.add_paragraph('This is a regular paragraph.')
    
    # Add heading
    doc.add_heading('Section 1', level=1)
    
    # Add paragraph with formatting
    p = doc.add_paragraph('This paragraph has ')
    p.add_run('bold').bold = True
    p.add_run(' and ')
    p.add_run('italic').italic = True
    p.add_run(' text.')
    
    # Add bullet list
    doc.add_paragraph('First item', style='List Bullet')
    doc.add_paragraph('Second item', style='List Bullet')
    doc.add_paragraph('Third item', style='List Bullet')
    
    # Add numbered list
    doc.add_paragraph('First step', style='List Number')
    doc.add_paragraph('Second step', style='List Number')
    
    # Save document
    doc.save(output_path)

create_simple_document('output.docx')
\`\`\`

### Creating Tables

\`\`\`python
from docx import Document
from docx.shared import Inches

def create_document_with_table(output_path: str):
    """Create document with formatted table."""
    doc = Document()
    
    doc.add_heading('Sales Report', 0)
    
    # Create table
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Light Grid Accent 1'
    
    # Header row
    header_cells = table.rows[0].cells
    header_cells[0].text = 'Product'
    header_cells[1].text = 'Quantity'
    header_cells[2].text = 'Price'
    
    # Data rows
    data = [
        ('Widget A', '100', '$1,050'),
        ('Widget B', '50', '$1,250'),
        ('Widget C', '75', '$1,181'),
    ]
    
    for product, qty, price in data:
        row_cells = table.add_row().cells
        row_cells[0].text = product
        row_cells[1].text = qty
        row_cells[2].text = price
    
    doc.save(output_path)

create_document_with_table('table_report.docx')
\`\`\`

### Advanced Formatting

\`\`\`python
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_formatted_document(output_path: str):
    """Create document with advanced formatting."""
    doc = Document()
    
    # Custom margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.25)
    
    # Styled title
    title = doc.add_heading('Professional Report', 0)
    title_format = title.runs[0].font
    title_format.size = Pt(24)
    title_format.color.rgb = RGBColor(0, 70, 127)  # Blue
    
    # Date with formatting
    date_para = doc.add_paragraph('Date: January 1, 2024')
    date_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    date_format = date_para.runs[0].font
    date_format.size = Pt(10)
    date_format.italic = True
    
    # Section with custom spacing
    section_heading = doc.add_heading('Executive Summary', level=1)
    
    # Paragraph with line spacing
    para = doc.add_paragraph('This is a paragraph with custom line spacing.')
    para_format = para.paragraph_format
    para_format.line_spacing = 1.5
    para_format.space_before = Pt(12)
    para_format.space_after = Pt(12)
    
    # Highlighted text
    highlight_para = doc.add_paragraph()
    highlight_run = highlight_para.add_run('Important: This text is highlighted.')
    highlight_run.font.highlight_color = True  # Yellow highlight
    highlight_run.bold = True
    
    doc.save(output_path)

create_formatted_document('formatted_report.docx')
\`\`\`

## Working with Templates

### Using Document Templates

\`\`\`python
from docx import Document
from typing import Dict

def fill_template(template_path: str, output_path: str, replacements: Dict[str, str]):
    """
    Fill a Word template by replacing placeholders.
    
    Template uses placeholders like {{name}}, {{date}}, etc.
    """
    doc = Document(template_path)
    
    # Replace in paragraphs
    for para in doc.paragraphs:
        for key, value in replacements.items():
            placeholder = f"{{{{{key}}}}}"
            if placeholder in para.text:
                # Replace in runs to preserve formatting
                for run in para.runs:
                    if placeholder in run.text:
                        run.text = run.text.replace(placeholder, value)
    
    # Replace in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for key, value in replacements.items():
                    placeholder = f"{{{{{key}}}}}"
                    if placeholder in cell.text:
                        cell.text = cell.text.replace(placeholder, value)
    
    doc.save(output_path)

# Usage: Generate contract from template
replacements = {
    "client_name": "Acme Corporation",
    "date": "January 15, 2024",
    "contract_value": "$50,000",
    "term": "12 months"
}

fill_template('contract_template.docx', 'filled_contract.docx', replacements)
\`\`\`

### Advanced Template Processing

\`\`\`python
from docx import Document
from typing import Dict, Any
import re

class WordTemplateProcessor:
    """
    Advanced template processor with conditional sections.
    
    Supports:
    - Variable replacement
    - Conditional sections ({{#if condition}}...{{/if}})
    - Loop sections ({{#each items}}...{{/each}})
    """
    
    def __init__(self, template_path: str):
        self.doc = Document(template_path)
    
    def fill(self, data: Dict[str, Any], output_path: str):
        """Fill template with data."""
        # Replace variables
        self._replace_variables(data)
        
        # Process conditional sections
        self._process_conditionals(data)
        
        # Save result
        self.doc.save(output_path)
    
    def _replace_variables(self, data: Dict[str, Any]):
        """Replace {{variable}} placeholders."""
        for para in self.doc.paragraphs:
            for run in para.runs:
                text = run.text
                
                # Find all {{variable}} patterns
                matches = re.findall(r'{{([a-zA-Z0-9_]+)}}', text)
                
                for var_name in matches:
                    if var_name in data:
                        placeholder = f"{{{{{var_name}}}}}"
                        value = str(data[var_name])
                        run.text = run.text.replace(placeholder, value)
    
    def _process_conditionals(self, data: Dict[str, Any]):
        """Process {{#if condition}}...{{/if}} blocks."""
        # Simplified implementation
        # Production version would handle full conditional logic
        pass

# Usage
processor = WordTemplateProcessor('report_template.docx')

data = {
    "company_name": "Tech Corp",
    "revenue": "$1.5M",
    "growth": "25%",
    "quarter": "Q4 2024"
}

processor.fill(data, 'quarterly_report.docx')
\`\`\`

## Adding Images

\`\`\`python
from docx import Document
from docx.shared import Inches

def add_images_to_document(output_path: str):
    """Create document with images."""
    doc = Document()
    
    doc.add_heading('Image Gallery', 0)
    
    # Add image with specific size
    doc.add_paragraph('Company Logo:')
    doc.add_picture('logo.png', width=Inches(2))
    
    # Add image with caption
    doc.add_paragraph()  # Spacer
    doc.add_paragraph('Chart showing results:')
    doc.add_picture('chart.png', width=Inches(4))
    caption = doc.add_paragraph('Figure 1: Sales Performance')
    caption.alignment = 1  # Center alignment
    
    doc.save(output_path)
\`\`\`

## Production Word Processor

\`\`\`python
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import logging

class WordDocumentProcessor:
    """
    Production-grade Word document processor.
    
    For building automated document generation systems.
    Foundation for document AI applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_document(self, filepath: str) -> Dict:
        """Read document with full structure extraction."""
        try:
            doc = Document(filepath)
            
            return {
                "filepath": filepath,
                "paragraphs": self._extract_paragraphs(doc),
                "tables": self._extract_tables(doc),
                "text": "\\n\\n".join(p.text for p in doc.paragraphs if p.text.strip())
            }
        except Exception as e:
            self.logger.error(f"Failed to read document: {e}")
            return {}
    
    def _extract_paragraphs(self, doc) -> List[Dict]:
        """Extract paragraphs with formatting."""
        paragraphs = []
        
        for para in doc.paragraphs:
            if not para.text.strip():
                continue
            
            paragraphs.append({
                "text": para.text,
                "style": para.style.name,
                "bold": any(run.bold for run in para.runs if run.bold),
                "italic": any(run.italic for run in para.runs if run.italic)
            })
        
        return paragraphs
    
    def _extract_tables(self, doc) -> List[pd.DataFrame]:
        """Extract tables as DataFrames."""
        tables = []
        
        for table in doc.tables:
            data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            
            if len(data) > 1:
                try:
                    df = pd.DataFrame(data[1:], columns=data[0])
                    tables.append(df)
                except:
                    pass
        
        return tables
    
    def generate_document(
        self,
        output_path: str,
        title: str,
        sections: List[Dict],
        add_toc: bool = False
    ) -> bool:
        """
        Generate professional document from structured content.
        
        Sections format:
        [
            {"type": "heading", "text": "Section Title", "level": 1},
            {"type": "paragraph", "text": "Content..."},
            {"type": "table", "data": DataFrame},
            {"type": "image", "path": "image.png", "width": 4}
        ]
        """
        try:
            doc = Document()
            
            # Add title
            if title:
                title_para = doc.add_heading(title, 0)
                title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add table of contents if requested
            if add_toc:
                doc.add_paragraph("Table of Contents")
                # Note: python-docx doesn't directly support TOC
                # Would need to use python-docx-template or manual XML
            
            # Add sections
            for section in sections:
                self._add_section(doc, section)
            
            doc.save(output_path)
            self.logger.info(f"Generated document: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate document: {e}")
            return False
    
    def _add_section(self, doc, section: Dict):
        """Add a section to the document."""
        section_type = section.get("type")
        
        if section_type == "heading":
            doc.add_heading(section["text"], level=section.get("level", 1))
        
        elif section_type == "paragraph":
            para = doc.add_paragraph(section["text"])
            if section.get("bold"):
                for run in para.runs:
                    run.bold = True
        
        elif section_type == "table" and "data" in section:
            self._add_table_from_dataframe(doc, section["data"])
        
        elif section_type == "image" and "path" in section:
            width = section.get("width", 4)
            try:
                doc.add_picture(section["path"], width=Inches(width))
            except:
                self.logger.warning(f"Failed to add image: {section['path']}")
        
        elif section_type == "list":
            for item in section.get("items", []):
                doc.add_paragraph(item, style='List Bullet')
    
    def _add_table_from_dataframe(self, doc, df: pd.DataFrame):
        """Add DataFrame as formatted table."""
        table = doc.add_table(rows=1, cols=len(df.columns))
        table.style = 'Light Grid Accent 1'
        
        # Header row
        header_cells = table.rows[0].cells
        for i, col_name in enumerate(df.columns):
            header_cells[i].text = str(col_name)
        
        # Data rows
        for _, row in df.iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = str(value)

# Usage Example: Generate report from LLM output
processor = WordDocumentProcessor()

# Read existing document for analysis
doc_data = processor.read_document("input.docx")
print(f"Document has {len(doc_data['paragraphs'])} paragraphs")

# Generate new document
sections = [
    {"type": "heading", "text": "Sales Analysis", "level": 1},
    {"type": "paragraph", "text": "This report analyzes Q4 2024 sales performance."},
    {"type": "heading", "text": "Key Metrics", "level": 2},
    {"type": "table", "data": pd.DataFrame({
        "Metric": ["Revenue", "Growth", "Customers"],
        "Value": ["$1.5M", "25%", "1,200"]
    })},
    {"type": "heading", "text": "Recommendations", "level": 2},
    {"type": "list", "items": [
        "Increase marketing budget",
        "Expand to new markets",
        "Improve customer retention"
    ]}
]

processor.generate_document("report.docx", "Q4 2024 Sales Report", sections)
\`\`\`

## Key Takeaways

1. **Use python-docx** for Word document manipulation
2. **Extract structure** not just text (paragraphs, tables, formatting)
3. **Use templates** for consistent document generation
4. **Preserve formatting** when reading and modifying
5. **Style with built-in styles** for professional appearance
6. **Extract tables** to pandas DataFrames for analysis
7. **Generate from structured data** for automation
8. **Handle images** for visual content
9. **Test with real documents** to handle edge cases
10. **Validate output** before delivering to users

These patterns enable building sophisticated document processing systems for LLM applications, from automated report generation to contract filling.`,
  videoUrl: undefined,
};
