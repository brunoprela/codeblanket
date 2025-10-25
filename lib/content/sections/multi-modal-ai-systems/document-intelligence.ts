export const documentIntelligence = {
  title: 'Document Intelligence',
  id: 'document-intelligence',
  description:
    'Master processing complex documents with mixed content - forms, invoices, receipts, reports - using multi-modal AI for extraction and understanding.',
  content: `
# Document Intelligence

## Introduction

Document intelligence involves extracting, understanding, and processing information from complex documents that contain mixed content: text, tables, images, forms, and various layouts. Traditional OCR falls short when dealing with modern documents that require semantic understanding.

In this section, we'll explore how to build systems that can intelligently process any document type, extract structured data, understand context, and handle real-world document complexity.

## Document Types and Challenges

### 1. Invoices

**Challenges:**
- Varied layouts across vendors
- Tables with line items
- Multiple currencies and tax calculations
- Headers and footers with important info
- Company logos and stamps

**Key Information:**
- Invoice number and date
- Vendor and customer information
- Line items with quantities and prices
- Subtotals, tax, and total amounts
- Payment terms

### 2. Receipts

**Challenges:**
- Often thermal printed (faded)
- Inconsistent formats
- Handwritten notes
- Crumpled or damaged
- Mixed languages

**Key Information:**
- Merchant name and location
- Date and time
- Items purchased
- Payment method
- Total amount

### 3. Forms

**Challenges:**
- Checkboxes and radio buttons
- Handwritten input
- Signatures
- Multi-page forms
- Conditional fields

**Key Information:**
- Field labels and values
- Checkbox/radio states
- Signatures present
- Form type and purpose

### 4. Reports

**Challenges:**
- Multi-page documents
- Charts and graphs
- Tables with complex data
- Headers and footers
- Mixed content types

**Key Information:**
- Executive summary
- Key metrics and KPIs
- Data tables
- Visual charts
- Conclusions and recommendations

## Building Document Intelligence Systems

### Basic Document Processing

\`\`\`python
import os
from typing import Dict, Any, List, Optional
from openai import OpenAI
import base64
from PIL import Image
import io

client = OpenAI()

def process_document_image(
    image_path: str,
    document_type: str = "generic",
    extraction_schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process document image and extract structured information.
    
    Args:
        image_path: Path to document image
        document_type: Type of document (invoice, receipt, form, report, etc.)
        extraction_schema: Optional JSON schema for extraction
    
    Returns:
        Extracted structured data
    """
    # Read and encode image
    with open (image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode (image_data).decode('utf-8')
    
    # Build extraction prompt based on document type
    if extraction_schema:
        prompt = f"""Extract information from this {document_type} according to this schema:

{json.dumps (extraction_schema, indent=2)}

Return the extracted data as JSON matching this schema. If a field is not found, use null."""
    
    else:
        # Default prompts by document type
        prompts = {
            "invoice": """Extract the following information from this invoice:

{{
  "invoice_number": "...",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "vendor": {{
    "name": "...",
    "address": "...",
    "contact": "..."
  }},
  "customer": {{
    "name": "...",
    "address": "..."
  }},
  "line_items": [
    {{
      "description": "...",
      "quantity": 0,
      "unit_price": 0.0,
      "amount": 0.0
    }}
  ],
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "currency": "USD"
}}

Be precise with numbers and dates.""",
            
            "receipt": """Extract information from this receipt:

{{
  "merchant_name": "...",
  "location": "...",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "items": [
    {{
      "description": "...",
      "quantity": 0,
      "price": 0.0
    }}
  ],
  "subtotal": 0.0,
  "tax": 0.0,
  "total": 0.0,
  "payment_method": "...",
  "currency": "USD"
}}""",
            
            "form": """Analyze this form and extract:

{{
  "form_type": "...",
  "fields": [
    {{
      "label": "...",
      "value": "...",
      "type": "text|checkbox|date|signature|..."
    }}
  ],
  "checkboxes": [
    {{
      "label": "...",
      "checked": true/false
    }}
  ],
  "has_signature": true/false
}}""",
            
            "generic": """Analyze this document and extract all visible text and structured information. Include:

{{
  "document_type": "...",
  "title": "...",
  "date": "...",
  "main_content": "...",
  "key_information": {{}}
}}"""
        }
        
        prompt = prompts.get (document_type, prompts["generic"])
    
    # Call Vision API
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"  # Use high detail for documents
                    }
                }
            ]
        }],
        max_tokens=2000,
        temperature=0.0  # Deterministic for data extraction
    )
    
    # Parse JSON response
    import json
    try:
        extracted_data = json.loads (response.choices[0].message.content)
    except json.JSONDecodeError:
        # Fallback: return raw response
        extracted_data = {
            "raw_response": response.choices[0].message.content,
            "extraction_failed": True
        }
    
    return extracted_data

# Example usage
invoice_data = process_document_image(
    "invoice.png",
    document_type="invoice"
)

print(f"Invoice Number: {invoice_data['invoice_number']}")
print(f"Total: \${invoice_data['total']}")
print(f"Line Items: {len (invoice_data['line_items'])}")
\`\`\`

### Production Document Intelligence System

\`\`\`python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import redis
import json
import logging
from pathlib import Path

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentExtractionResult:
    """Result of document extraction."""
    document_id: str
    document_type: str
    extracted_data: Dict[str, Any]
    confidence: float
    processing_time: float
    validation_errors: List[str]
    cached: bool = False

class ProductionDocumentIntelligence:
    """Production-ready document intelligence system."""
    
    def __init__(
        self,
        openai_api_key: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 86400 * 30  # 30 days
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.redis_client = redis.Redis (host=redis_host, port=redis_port)
        self.cache_ttl = cache_ttl
        
        # Define extraction schemas for different document types
        self.schemas = self._load_schemas()
    
    def _load_schemas (self) -> Dict[str, Dict]:
        """Load document extraction schemas."""
        return {
            "invoice": {
                "required_fields": [
                    "invoice_number",
                    "invoice_date",
                    "vendor",
                    "total"
                ],
                "numeric_fields": ["total", "subtotal", "tax"],
                "date_fields": ["invoice_date", "due_date"]
            },
            "receipt": {
                "required_fields": [
                    "merchant_name",
                    "date",
                    "total"
                ],
                "numeric_fields": ["total", "subtotal", "tax"],
                "date_fields": ["date"]
            },
            "form": {
                "required_fields": ["form_type", "fields"],
                "numeric_fields": [],
                "date_fields": []
            }
        }
    
    def _get_document_hash (self, image_path: str) -> str:
        """Generate hash of document image."""
        with open (image_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _get_cache_key (self, doc_hash: str, doc_type: str) -> str:
        """Generate cache key."""
        return f"doc_intel:{doc_hash}:{doc_type}"
    
    def _preprocess_image(
        self,
        image_path: str,
        enhance: bool = True
    ) -> str:
        """
        Preprocess document image for better extraction.
        
        Args:
            image_path: Path to image
            enhance: Whether to enhance image quality
        
        Returns:
            Path to preprocessed image
        """
        from PIL import Image, ImageEnhance, ImageFilter
        
        img = Image.open (image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if enhance:
            # Increase contrast
            enhancer = ImageEnhance.Contrast (img)
            img = enhancer.enhance(1.5)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness (img)
            img = enhancer.enhance(1.5)
            
            # Apply slight denoise
            img = img.filter(ImageFilter.MedianFilter (size=3))
        
        # Save preprocessed image
        output_path = "preprocessed_" + Path (image_path).name
        img.save (output_path, 'JPEG', quality=95)
        
        return output_path
    
    def _validate_extraction(
        self,
        extracted_data: Dict[str, Any],
        document_type: str
    ) -> List[str]:
        """
        Validate extracted data against schema.
        
        Returns list of validation errors.
        """
        errors = []
        
        if document_type not in self.schemas:
            return errors
        
        schema = self.schemas[document_type]
        
        # Check required fields
        for field in schema["required_fields"]:
            if field not in extracted_data or not extracted_data[field]:
                errors.append (f"Missing required field: {field}")
        
        # Validate numeric fields
        for field in schema["numeric_fields"]:
            if field in extracted_data:
                try:
                    float (extracted_data[field])
                except (ValueError, TypeError):
                    errors.append (f"Invalid numeric value for field: {field}")
        
        # Validate date fields
        for field in schema["date_fields"]:
            if field in extracted_data and extracted_data[field]:
                try:
                    datetime.strptime (extracted_data[field], "%Y-%m-%d")
                except ValueError:
                    errors.append (f"Invalid date format for field: {field}")
        
        return errors
    
    def _calculate_confidence(
        self,
        extracted_data: Dict[str, Any],
        validation_errors: List[str]
    ) -> float:
        """Calculate confidence score for extraction."""
        # Start with base confidence
        confidence = 1.0
        
        # Reduce confidence for each validation error
        confidence -= len (validation_errors) * 0.1
        
        # Reduce confidence for missing optional fields
        # (Implementation depends on schema)
        
        return max(0.0, min(1.0, confidence))
    
    def process_document(
        self,
        image_path: str,
        document_type: str,
        use_cache: bool = True,
        enhance_image: bool = True,
        custom_schema: Optional[Dict] = None
    ) -> DocumentExtractionResult:
        """
        Process document and extract structured data.
        
        Args:
            image_path: Path to document image
            document_type: Type of document
            use_cache: Whether to use cache
            enhance_image: Whether to enhance image quality
            custom_schema: Optional custom extraction schema
        
        Returns:
            DocumentExtractionResult
        """
        start_time = datetime.now()
        
        # Generate document ID
        doc_hash = self._get_document_hash (image_path)
        doc_id = f"{document_type}_{doc_hash[:8]}"
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key (doc_hash, document_type)
            cached_result = self.redis_client.get (cache_key)
            
            if cached_result:
                logger.info (f"Cache hit for document {doc_id}")
                result_dict = json.loads (cached_result)
                return DocumentExtractionResult(**result_dict, cached=True)
        
        # Preprocess image
        if enhance_image:
            processed_image_path = self._preprocess_image (image_path)
        else:
            processed_image_path = image_path
        
        # Extract data
        extracted_data = process_document_image(
            processed_image_path,
            document_type,
            custom_schema
        )
        
        # Validate extraction
        validation_errors = self._validate_extraction (extracted_data, document_type)
        
        # Calculate confidence
        confidence = self._calculate_confidence (extracted_data, validation_errors)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = DocumentExtractionResult(
            document_id=doc_id,
            document_type=document_type,
            extracted_data=extracted_data,
            confidence=confidence,
            processing_time=processing_time,
            validation_errors=validation_errors,
            cached=False
        )
        
        # Cache result
        if use_cache:
            result_dict = {
                "document_id": result.document_id,
                "document_type": result.document_type,
                "extracted_data": result.extracted_data,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "validation_errors": result.validation_errors
            }
            
            cache_key = self._get_cache_key (doc_hash, document_type)
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps (result_dict)
            )
        
        logger.info(
            f"Processed document {doc_id} in {processing_time:.2f}s "
            f"(confidence: {confidence:.2f})"
        )
        
        return result
    
    def batch_process_documents(
        self,
        documents: List[Dict[str, str]],
        max_concurrent: int = 5
    ) -> List[DocumentExtractionResult]:
        """
        Process multiple documents concurrently.
        
        Args:
            documents: List of dicts with 'path' and 'type' keys
            max_concurrent: Maximum concurrent processes
        
        Returns:
            List of extraction results
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def process_one (doc: Dict[str, str]) -> DocumentExtractionResult:
            return self.process_document (doc['path'], doc['type'])
        
        with ThreadPoolExecutor (max_workers=max_concurrent) as executor:
            results = list (executor.map (process_one, documents))
        
        return results

# Usage
doc_intel = ProductionDocumentIntelligence(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

result = doc_intel.process_document(
    "invoice.png",
    document_type="invoice",
    enhance_image=True
)

print(f"Document ID: {result.document_id}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Processing Time: {result.processing_time:.2f}s")
print(f"\\nExtracted Data:")
print(json.dumps (result.extracted_data, indent=2))

if result.validation_errors:
    print(f"\\nValidation Errors:")
    for error in result.validation_errors:
        print(f"  - {error}")
\`\`\`

### Table Extraction

\`\`\`python
def extract_table_from_document(
    image_path: str,
    table_description: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract table data from document image.
    
    Args:
        image_path: Path to document with table
        table_description: Optional description of what table to extract
    
    Returns:
        List of dictionaries representing table rows
    """
    with open (image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode (image_data).decode('utf-8')
    
    prompt = """Extract the table from this document.

Return the data as a JSON array where each element is a row:
[
  {"column1": "value", "column2": "value", ...},
  ...
]

Use the first row as column headers. Be precise with numbers and text."""

    if table_description:
        prompt += f"\\n\\nTable to extract: {table_description}"
    
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }],
        max_tokens=2000,
        temperature=0.0
    )
    
    import json
    table_data = json.loads (response.choices[0].message.content)
    
    return table_data

# Extract table
table = extract_table_from_document(
    "report_with_table.png",
    table_description="quarterly revenue table"
)

print(f"Extracted {len (table)} rows")
for row in table[:3]:
    print(row)
\`\`\`

### Form Processing

\`\`\`python
def process_filled_form(
    image_path: str,
    form_template: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Any]:
    """
    Process filled form and extract field values.
    
    Args:
        image_path: Path to filled form image
        form_template: Optional template with expected fields
    
    Returns:
        Extracted form data
    """
    with open (image_path, "rb") as f:
        image_data = f.read()
    
    base64_image = base64.b64encode (image_data).decode('utf-8')
    
    if form_template:
        field_list = "\\n".join([f"- {field}" for field in form_template.get("fields", [])])
        prompt = f"""Extract information from this filled form. Expected fields:

{field_list}

Return as JSON:
{{
  "fields": {{
    "field_name": "value"
  }},
  "checkboxes": {{
    "checkbox_label": true/false
  }},
  "signatures_present": true/false
}}"""
    else:
        prompt = """Extract all information from this filled form.

Return as JSON with:
- All field labels and their values
- Checkbox states
- Whether signatures are present
- Any additional notes or annotations"""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }],
        max_tokens=1500,
        temperature=0.0
    )
    
    import json
    form_data = json.loads (response.choices[0].message.content)
    
    return form_data

# Process form
form_data = process_filled_form("application_form.png")
print(json.dumps (form_data, indent=2))
\`\`\`

### Multi-Page Document Processing

\`\`\`python
import fitz  # PyMuPDF

def process_multi_page_pdf(
    pdf_path: str,
    document_type: str = "report"
) -> Dict[str, Any]:
    """
    Process multi-page PDF document.
    
    Args:
        pdf_path: Path to PDF file
        document_type: Type of document
    
    Returns:
        Aggregated extracted data
    """
    doc = fitz.open (pdf_path)
    
    page_data = []
    
    for page_num, page in enumerate (doc):
        # Convert page to image
        pix = page.get_pixmap (matrix=fitz.Matrix(2, 2))  # 2x resolution
        img_data = pix.tobytes("png")
        
        # Save page as image
        page_image_path = f"page_{page_num + 1}.png"
        with open (page_image_path, "wb") as f:
            f.write (img_data)
        
        # Extract data from page
        page_result = process_document_image(
            page_image_path,
            document_type
        )
        
        page_data.append({
            "page": page_num + 1,
            "data": page_result
        })
    
    doc.close()
    
    # Aggregate data from all pages
    aggregated = {
        "document_type": document_type,
        "total_pages": len (page_data),
        "pages": page_data,
        "combined_text": " ".join([
            str (p["data"].get("main_content", ""))
            for p in page_data
        ])
    }
    
    return aggregated

# Process PDF
result = process_multi_page_pdf("report.pdf", document_type="report")
print(f"Processed {result['total_pages']} pages")
\`\`\`

## Advanced Techniques

### Confidence Scoring

\`\`\`python
def calculate_field_confidence(
    field_value: str,
    field_type: str,
    validation_rules: Optional[Dict] = None
) -> float:
    """Calculate confidence score for extracted field."""
    confidence = 1.0
    
    # Check if field is empty
    if not field_value:
        return 0.0
    
    # Validate based on field type
    if field_type == "date":
        try:
            datetime.strptime (field_value, "%Y-%m-%d")
        except ValueError:
            confidence *= 0.5
    
    elif field_type == "number":
        try:
            float (field_value)
        except ValueError:
            confidence *= 0.3
    
    elif field_type == "email":
        import re
        if not re.match (r"[^@]+@[^@]+\\.[^@]+", field_value):
            confidence *= 0.4
    
    # Apply custom validation rules
    if validation_rules:
        for rule_name, rule_func in validation_rules.items():
            if not rule_func (field_value):
                confidence *= 0.7
    
    return confidence
\`\`\`

### Error Correction

\`\`\`python
def correct_extraction_errors(
    extracted_data: Dict[str, Any],
    document_type: str
) -> Dict[str, Any]:
    """
    Attempt to correct common extraction errors.
    
    Args:
        extracted_data: Extracted data
        document_type: Document type
    
    Returns:
        Corrected data
    """
    corrected = extracted_data.copy()
    
    # Correct common OCR errors in numbers
    if "total" in corrected and isinstance (corrected["total"], str):
        # Remove currency symbols
        total = corrected["total"].replace("$", "").replace("€", "")
        # Remove commas
        total = total.replace(",", "")
        # Convert to float
        try:
            corrected["total"] = float (total)
        except ValueError:
            pass
    
    # Correct date formats
    if "date" in corrected:
        date_str = corrected["date"]
        # Try multiple date formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime (date_str, fmt)
                corrected["date"] = dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
    
    # Normalize field names
    # (e.g., "invoice_no" → "invoice_number")
    field_mappings = {
        "invoice_no": "invoice_number",
        "inv_date": "invoice_date",
        "amt": "amount"
    }
    
    for old_key, new_key in field_mappings.items():
        if old_key in corrected and new_key not in corrected:
            corrected[new_key] = corrected.pop (old_key)
    
    return corrected
\`\`\`

### Human-in-the-Loop Validation

\`\`\`python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationRequest:
    """Request for human validation."""
    document_id: str
    extracted_data: Dict[str, Any]
    confidence: float
    validation_errors: List[str]
    image_path: str
    created_at: datetime

def needs_human_validation(
    result: DocumentExtractionResult,
    confidence_threshold: float = 0.8
) -> bool:
    """Determine if extraction needs human validation."""
    # Low confidence
    if result.confidence < confidence_threshold:
        return True
    
    # Has validation errors
    if result.validation_errors:
        return True
    
    # Critical document types always need validation
    critical_types = ["contract", "legal", "financial"]
    if result.document_type in critical_types:
        return True
    
    return False

def create_validation_request(
    result: DocumentExtractionResult,
    image_path: str
) -> ValidationRequest:
    """Create a human validation request."""
    return ValidationRequest(
        document_id=result.document_id,
        extracted_data=result.extracted_data,
        confidence=result.confidence,
        validation_errors=result.validation_errors,
        image_path=image_path,
        created_at=datetime.now()
    )

# Example usage
result = doc_intel.process_document("important_invoice.png", "invoice")

if needs_human_validation (result):
    validation_request = create_validation_request (result, "important_invoice.png")
    print(f"Document {result.document_id} requires human validation")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Errors: {len (result.validation_errors)}")
    # Send to validation queue
else:
    print(f"Document {result.document_id} passed automatic validation")
    # Process automatically
\`\`\`

## Real-World Applications

### 1. Invoice Processing System

\`\`\`python
def build_invoice_processing_system() -> Dict[str, Any]:
    """Complete invoice processing system."""
    doc_intel = ProductionDocumentIntelligence(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    def process_invoice (invoice_path: str) -> Dict[str, Any]:
        # Process invoice
        result = doc_intel.process_document (invoice_path, "invoice")
        
        # Correct common errors
        corrected_data = correct_extraction_errors(
            result.extracted_data,
            "invoice"
        )
        
        # Calculate line item totals for validation
        if "line_items" in corrected_data:
            calculated_subtotal = sum(
                item.get("amount", 0)
                for item in corrected_data["line_items"]
            )
            
            # Check if it matches extracted subtotal
            extracted_subtotal = corrected_data.get("subtotal", 0)
            if abs (calculated_subtotal - extracted_subtotal) > 0.01:
                result.validation_errors.append(
                    f"Line items total ({calculated_subtotal}) "
                    f"doesn't match subtotal ({extracted_subtotal})"
                )
        
        # Determine next action
        if needs_human_validation (result):
            return {
                "status": "needs_validation",
                "data": corrected_data,
                "validation_request": create_validation_request (result, invoice_path)
            }
        else:
            return {
                "status": "approved",
                "data": corrected_data,
                "confidence": result.confidence
            }
    
    return process_invoice

# Usage
process_invoice = build_invoice_processing_system()
result = process_invoice("vendor_invoice.png")

if result["status"] == "approved":
    print("Invoice automatically approved")
    # Send to accounting system
else:
    print("Invoice needs human review")
    # Send to validation queue
\`\`\`

### 2. Expense Report Processing

\`\`\`python
def process_expense_report(
    receipts: List[str],
    report_form: Optional[str] = None
) -> Dict[str, Any]:
    """Process expense report with multiple receipts."""
    doc_intel = ProductionDocumentIntelligence(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Process all receipts
    receipt_data = []
    total_amount = 0.0
    
    for receipt_path in receipts:
        result = doc_intel.process_document (receipt_path, "receipt")
        
        if result.confidence > 0.7:
            receipt_data.append (result.extracted_data)
            total_amount += result.extracted_data.get("total", 0.0)
    
    # Process report form if provided
    form_data = None
    if report_form:
        form_result = doc_intel.process_document (report_form, "form")
        form_data = form_result.extracted_data
    
    return {
        "receipts": receipt_data,
        "total_receipts": len (receipt_data),
        "total_amount": total_amount,
        "form_data": form_data,
        "status": "complete"
    }
\`\`\`

## Best Practices

### 1. Image Quality

- Use high resolution (300+ DPI for scanned documents)
- Ensure good lighting and no shadows
- Straighten skewed documents
- Remove noise and artifacts
- Use high detail level in API

### 2. Validation

- Always validate extracted data
- Implement field-specific validation rules
- Check for required fields
- Validate data types and formats
- Compare calculated vs extracted totals

### 3. Error Handling

- Implement retry logic
- Have fallback parsing strategies
- Log extraction failures
- Queue low-confidence extractions for review
- Provide clear error messages

### 4. Cost Optimization

- Cache results by document hash
- Use image preprocessing to improve accuracy
- Batch process when possible
- Monitor API usage
- Implement confidence-based routing (high confidence = auto-process)

## Summary

Document intelligence enables automated processing of complex documents:

**Key Capabilities:**
- Invoice and receipt processing
- Form extraction
- Table extraction from reports
- Multi-page document handling
- Handwriting recognition
- Mixed content understanding

**Production Patterns:**
- Preprocess images for better quality
- Implement comprehensive validation
- Use confidence scoring
- Human-in-the-loop for critical documents
- Cache results aggressively
- Batch processing for efficiency

**Best Practices:**
- Use high detail level for documents
- Validate extracted data programmatically
- Implement error correction
- Monitor confidence scores
- Queue low-confidence for human review
- Log everything for debugging

**Applications:**
- Accounts payable automation
- Expense report processing
- Form digitization
- Contract analysis
- Medical record processing
- Insurance claim processing

Next, we'll explore presentation and slide generation from content.
`,
  codeExamples: [
    {
      title: 'Production Document Intelligence System',
      description:
        'Complete document processing system with validation, caching, and error handling',
      language: 'python',
      code: `# See ProductionDocumentIntelligence class in content above`,
    },
  ],
  practicalTips: [
    "Always use 'high' detail level for document images - accuracy is critical",
    'Preprocess images: straighten, enhance contrast, denoise for better extraction',
    'Implement field-specific validation rules (dates, numbers, emails, etc.)',
    'Cache extraction results by document hash - documents rarely change',
    'Set confidence thresholds: >0.9 auto-process, 0.7-0.9 review, <0.7 manual entry',
    'Always validate calculated totals against extracted totals for invoices/receipts',
    'Use temperature=0.0 for deterministic data extraction',
    'Implement human-in-the-loop for critical document types (contracts, legal, etc.)',
  ],
  quiz: '/quizzes/multi-modal-ai-systems/document-intelligence',
};
