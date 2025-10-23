/**
 * Quiz questions for Word Document Processing section
 */

export const worddocumentprocessingQuiz = [
  {
    id: 'fpdu-word-proc-q-1',
    question:
      'Design an automated contract generation system that fills Word templates based on data from a database. How would you handle conditional sections, dynamic tables, and formatting preservation?',
    hint: 'Consider template design, placeholder replacement, conditional logic, and validation.',
    sampleAnswer:
      'Contract generation system design: (1) Template design: create Word template with placeholders ({{client_name}}, {{contract_value}}, etc.), mark conditional sections with special syntax ({{#if condition}}...{{/if}}), use table templates for dynamic data. (2) Data extraction: query database for contract data, validate all required fields are present, prepare data structure matching template expectations. (3) Template processing: load template with python-docx, replace simple placeholders in paragraphs and tables (preserve run formatting), evaluate conditional sections and remove/keep based on data, populate dynamic tables from data arrays. (4) Formatting preservation: use run-level replacement not paragraph-level to keep formatting, maintain styles for headings and body text, preserve spacing and margins. (5) Validation: check all placeholders were replaced, verify no {{}} remain, validate required sections are present, check generated document opens correctly. (6) Review workflow: generate PDF preview for human review, allow corrections before final generation, log all generations for audit. Production considerations: cache template parsing, handle errors gracefully with detailed messages, support multiple template versions.',
    keyPoints: [
      'Use {{placeholder}} syntax for variable replacement',
      'Replace at run level to preserve formatting',
      'Implement conditional sections with custom syntax',
      'Populate dynamic tables from data arrays',
      'Validate all placeholders are replaced',
      'Generate PDF preview for human review',
      'Log all generations for audit trail',
    ],
  },
  {
    id: 'fpdu-word-proc-q-2',
    question:
      'How would you build a system that analyzes Word documents to extract structured data (tables, sections, metadata) and convert them to a format suitable for LLM processing?',
    hint: 'Think about structure detection, data normalization, chunking, and preserving relationships.',
    sampleAnswer:
      'Document analysis pipeline: (1) Structure extraction: read with python-docx, identify document structure (headings, paragraphs, tables), detect heading hierarchy (H1, H2, H3) for section boundaries, extract formatting (bold, italic) as semantic markers. (2) Table processing: extract all tables to pandas DataFrames, clean and normalize data, detect header rows and data types, maintain table context (which section contains it). (3) Section segmentation: group paragraphs under their parent headings, maintain hierarchical relationships, preserve section metadata. (4) Metadata extraction: extract title, author, creation date, identify key-value pairs in content, detect lists and enumerated items. (5) LLM preparation: convert to structured JSON or markdown, chunk by section for token limits, include table data in readable format (markdown tables or JSON), provide document outline for context. (6) Relationship preservation: link references between sections, maintain table/figure references, preserve cross-references. (7) Validation: ensure no data loss during extraction, verify structure makes sense, check chunk sizes for LLM limits. Use case: extract contracts, identify clauses, prepare for legal analysis.',
    keyPoints: [
      'Extract structure: headings, paragraphs, tables, formatting',
      'Maintain hierarchical relationships between sections',
      'Convert tables to DataFrames for structured processing',
      'Chunk by sections while preserving context',
      'Convert to LLM-friendly format (markdown or JSON)',
      'Preserve cross-references and relationships',
      'Validate extraction completeness and accuracy',
    ],
  },
  {
    id: 'fpdu-word-proc-q-3',
    question:
      'Compare generating Word documents with python-docx versus using template libraries like python-docx-template or docxtpl. When would you use each approach?',
    hint: 'Consider complexity, flexibility, maintainability, and non-technical user involvement.',
    sampleAnswer:
      'Approach comparison: (1) python-docx (programmatic): Build document from scratch using code, full control over structure and formatting, good for dynamic layouts and complex logic, requires code changes for template modifications. Use when: document structure varies significantly, need complex conditional logic, templates are code-managed, developers maintain templates. Pros: maximum flexibility, no template file needed, easy to version control code. Cons: harder for non-developers to modify, coupling between data and presentation. (2) python-docx-template / docxtpl (template-based): Non-technical users can design templates in Word, use Jinja2 syntax in Word documents ({{variable}}, {% if %}, {% for %}), separate presentation from logic, easier to maintain visual design. Use when: business users design templates, need quick template iterations, consistent document structure, non-developers maintain templates. Pros: easier for non-technical users, rapid template changes, visual template design. Cons: limited dynamic capabilities, learning curve for Jinja2 syntax. Best practice: hybrid approach - use templates for standard documents, programmatic for complex dynamic documents, provide template designer tool for business users.',
    keyPoints: [
      'python-docx: programmatic, full control, code-maintained',
      'Template libraries: Word-designed, business user friendly',
      'python-docx for complex dynamic layouts',
      'Templates for standard documents with variations',
      'Template libraries separate presentation from logic',
      'python-docx easier to version control',
      'Hybrid: templates for standard, code for complex',
    ],
  },
];
