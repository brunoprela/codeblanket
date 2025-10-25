/**
 * Output Format Control Section
 * Module 2: Prompt Engineering & Optimization
 */

export const outputformatcontrolSection = {
  id: 'output-format-control',
  title: 'Output Format Control',
  content: `# Output Format Control

Master techniques to enforce specific output formats, making LLM outputs reliable and parseable in production systems.

## Overview: Why Format Matters

Unstructured LLM outputs are hard to use in code. **Structured, predictable formats** make outputs easy to parse, validate, and integrate.

### The Problem

\`\`\`python
# Unreliable: Format varies each time
prompt = "Extract name and age from: 'John is 25 years old'"

# Responses vary:
# "John, 25"
# "Name: John, Age: 25" 
# "The name is John and the age is 25"
# Each response needs different parsing logic!

# Solution: Enforce format
prompt = """Extract name and age as JSON:
{
  "name": "...",
  "age": ...
}

Text: 'John is 25 years old'
Output:"""

# Now always get: {"name": "John", "age": 25}
# Easy to parse!
\`\`\`

## JSON Output Specification

### The Gold Standard for Structured Data

\`\`\`python
from openai import OpenAI
import json

client = OpenAI()

def extract_structured_data (text: str) -> dict:
    """Extract data in guaranteed JSON format."""
    
    prompt = f"""Extract person information as JSON with these exact fields:
{{
  "name": "full name",
  "age": number or null,
  "occupation": "job title or null"
}}

Text: {text}

Output only valid JSON:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}  # GPT-4 JSON mode
    )
    
    # Parse JSON
    return json.loads (response.choices[0].message.content)

# Test
result = extract_structured_data("Sarah Johnson is a 30-year-old software engineer")
print(result)
# {'name': 'Sarah Johnson', 'age': 30, 'occupation': 'software engineer'}

# Type-safe access
print(f"{result['name']} is {result['age']} years old")
\`\`\`

### JSON Schema Enforcement

\`\`\`python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import json

class Person(BaseModel):
    """Structured person data."""
    name: str = Field(..., description="Full name")
    age: Optional[int] = Field(None, description="Age in years")
    email: Optional[str] = Field(None, description="Email address")
    skills: List[str] = Field (default_factory=list, description="List of skills")

def create_json_schema_prompt (schema_model: type[BaseModel]) -> str:
    """
    Generate prompt with JSON schema from Pydantic model.
    Ensures type-safe, validated outputs.
    """
    
    # Get JSON schema from Pydantic
    schema = schema_model.model_json_schema()
    
    # Create example based on schema
    example = {}
    for field_name, field_info in schema.get('properties', {}).items():
        field_type = field_info.get('type', 'string')
        if field_type == 'string':
            example[field_name] = f"<{field_name}>"
        elif field_type == 'integer':
            example[field_name] = 0
        elif field_type == 'array':
            example[field_name] = []
    
    schema_str = json.dumps (schema, indent=2)
    example_str = json.dumps (example, indent=2)
    
    return f"""Extract information matching this JSON schema:

Schema:
{schema_str}

Example format:
{example_str}

Output only valid JSON matching the schema."""

# Generate prompt
prompt_template = create_json_schema_prompt(Person)

def extract_with_schema (text: str) -> Person:
    """Extract data with schema validation."""
    
    full_prompt = f"""{prompt_template}

Text: {text}

JSON:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"}
    )
    
    data = json.loads (response.choices[0].message.content)
    
    # Validate with Pydantic
    return Person(**data)

# Usage
person = extract_with_schema(
    "Alice Smith, age 28, alice@example.com, knows Python, JavaScript, and SQL"
)

print(person.model_dump())
# Type-safe access
print(f"Skills: {', '.join (person.skills)}")
\`\`\`

## Instructor Library: Type-Safe LLM Outputs

### The Best Way to Get Structured Outputs

\`\`\`python
"""
Instructor makes it trivial to get structured outputs.
It wraps OpenAI API with Pydantic validation.

Install: pip install instructor
"""

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Patch OpenAI client
client = instructor.from_openai(OpenAI())

class CodeReview(BaseModel):
    """Structured code review output."""
    summary: str = Field(..., description="One-sentence summary")
    bugs: List[str] = Field (default_factory=list, description="List of bugs found")
    suggestions: List[str] = Field (default_factory=list, description="Improvement suggestions")
    severity: str = Field(..., description="Overall severity: low/medium/high")
    code_quality_score: int = Field(..., ge=0, le=10, description="Quality score 0-10")

def review_code_structured (code: str) -> CodeReview:
    """Get structured code review."""
    
    review = client.chat.completions.create(
        model="gpt-4",
        response_model=CodeReview,  # Magic happens here!
        messages=[
            {
                "role": "user",
                "content": f"Review this code:\\n\\n{code}"
            }
        ]
    )
    
    return review

# Usage
code = """
def calculate_average (numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len (numbers)
"""

review = review_code_structured (code)

# Type-safe access with autocomplete!
print(f"Summary: {review.summary}")
print(f"Bugs: {len (review.bugs)}")
print(f"Quality: {review.code_quality_score}/10")
print(f"Severity: {review.severity}")

# Pydantic validation ensures correct types
assert isinstance (review.code_quality_score, int)
assert 0 <= review.code_quality_score <= 10
\`\`\`

### Complex Nested Structures

\`\`\`python
from typing import List, Optional
from pydantic import BaseModel, Field

class FunctionCall(BaseModel):
    """A function being called in code."""
    name: str
    arguments: List[str]
    return_type: Optional[str] = None

class CodeAnalysis(BaseModel):
    """Complete code analysis with nested structures."""
    language: str
    functions_defined: List[str]
    functions_called: List[FunctionCall]
    imports: List[str]
    complexity_score: int = Field(..., ge=1, le=10)
    recommendations: List[str]

def analyze_code (code: str) -> CodeAnalysis:
    """Analyze code and return structured analysis."""
    
    analysis = client.chat.completions.create(
        model="gpt-4",
        response_model=CodeAnalysis,
        messages=[
            {
                "role": "system",
                "content": "You are a code analyzer. Extract detailed information."
            },
            {
                "role": "user",
                "content": f"Analyze this code:\\n\\n{code}"
            }
        ]
    )
    
    return analysis

# Example
code = """
import requests
from typing import List

def fetch_users (api_key: str) -> List[dict]:
    response = requests.get('https://api.example.com/users', 
                           headers={'Authorization': api_key})
    return response.json()

def process_users (users: List[dict]) -> None:
    for user in users:
        print(user['name'])
"""

analysis = analyze_code (code)

print(f"Language: {analysis.language}")
print(f"Functions: {', '.join (analysis.functions_defined)}")
print(f"\\nFunction Calls:")
for call in analysis.functions_called:
    print(f"  - {call.name}({', '.join (call.arguments)})")
print(f"\\nComplexity: {analysis.complexity_score}/10")
\`\`\`

## XML and Markdown Formats

### Structured Alternative to JSON

\`\`\`python
def xml_format_prompt (task: str, input_data: str) -> str:
    """
    Some tasks work better with XML than JSON.
    Especially for hierarchical or mixed content.
    """
    
    return f"""{task}

Format your response as XML with this structure:
<response>
  <summary>Brief summary here</summary>
  <details>
    <point>First point</point>
    <point>Second point</point>
  </details>
  <conclusion>Final thoughts</conclusion>
</response>

Input: {input_data}

Response:"""

def parse_xml_response (xml_str: str) -> dict:
    """Parse XML response."""
    import xml.etree.ElementTree as ET
    
    root = ET.fromstring (xml_str)
    
    return {
        'summary': root.find('summary').text,
        'details': [point.text for point in root.findall('.//point')],
        'conclusion': root.find('conclusion').text
    }

# Usage
prompt = xml_format_prompt(
    task="Analyze this business proposal",
    input_data="Launch new product line targeting millennials..."
)

# Get response (simulated)
xml_response = """<response>
  <summary>Promising opportunity with moderate risk</summary>
  <details>
    <point>Strong market demand in target demographic</point>
    <point>Competitive pricing needed</point>
    <point>Marketing strategy requires refinement</point>
  </details>
  <conclusion>Recommend proceeding with pilot launch</conclusion>
</response>"""

parsed = parse_xml_response (xml_response)
print("Summary:", parsed['summary'])
print("Details:", parsed['details'])
\`\`\`

### Markdown for Human-Readable Structured Output

\`\`\`python
def markdown_format_prompt (task: str) -> str:
    """
    Markdown provides structure that's both machine-parseable
    and human-readable.
    """
    
    return f"""{task}

Format as Markdown with this exact structure:

## Summary
[One paragraph summary]

## Key Points
- Point 1
- Point 2
- Point 3

## Recommendations
1. First recommendation
2. Second recommendation

## Code Example (if applicable)
\`\`\`python
# code here
\`\`\`

Generate report:"""

def parse_markdown_sections (md_text: str) -> dict:
    """Parse markdown into structured sections."""
    
    sections = {}
    current_section = None
    current_content = []
    
    for line in md_text.split('\\n'):
        if line.startswith('## '):
            # Save previous section
            if current_section:
                sections[current_section] = '\\n'.join (current_content).strip()
            # Start new section
            current_section = line[3:].strip().lower().replace(' ', '_')
            current_content = []
        else:
            current_content.append (line)
    
    # Save last section
    if current_section:
        sections[current_section] = '\\n'.join (current_content).strip()
    
    return sections

# Example
md_output = """## Summary
The proposed architecture is scalable and follows best practices.

## Key Points
- Microservices design enables independent scaling
- API gateway provides centralized security
- Event-driven communication reduces coupling

## Recommendations
1. Implement circuit breakers for resilience
2. Add comprehensive monitoring
3. Plan for data consistency patterns"""

parsed = parse_markdown_sections (md_output)
print("Summary:", parsed['summary'])
print("\\nKey Points:", parsed['key_points'])
\`\`\`

## Schema Enforcement with Validation

### Ensuring Output Matches Specification

\`\`\`python
from typing import Any, Dict
from pydantic import BaseModel, ValidationError
import json

class OutputValidator:
    """
    Validate LLM outputs match expected schema.
    Retry with corrections if validation fails.
    """
    
    def __init__(self, client, model: str = "gpt-4"):
        self.client = client
        self.model = model
        self.max_retries = 3
    
    def get_validated_output(
        self,
        prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.3
    ) -> BaseModel:
        """
        Get LLM output and validate against schema.
        Retry with error feedback if validation fails.
        """
        
        for attempt in range (self.max_retries):
            try:
                # Get response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=temperature
                )
                
                output_text = response.choices[0].message.content
                
                # Parse JSON
                data = json.loads (output_text)
                
                # Validate against schema
                validated = schema(**data)
                
                print(f"✓ Validation successful (attempt {attempt + 1})")
                return validated
                
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"✗ Validation failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries - 1:
                    # Retry with error feedback
                    error_msg = str (e)
                    prompt = f"""{prompt}

Previous attempt failed validation:
Error: {error_msg}

Please correct the output to match the schema exactly."""
                else:
                    raise ValueError (f"Failed to get valid output after {self.max_retries} attempts")
    
    def validate_and_fix(
        self,
        output: str,
        schema: type[BaseModel]
    ) -> BaseModel:
        """
        Validate output and ask LLM to fix if invalid.
        """
        
        try:
            data = json.loads (output)
            return schema(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Ask LLM to fix
            fix_prompt = f"""This JSON output failed validation:

Output:
{output}

Error:
{e}

Schema required:
{schema.model_json_schema()}

Please provide corrected JSON that passes validation."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": fix_prompt}],
                response_format={"type": "json_object"}
            )
            
            fixed_output = response.choices[0].message.content
            data = json.loads (fixed_output)
            return schema(**data)

# Example
class Product(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    in_stock: bool
    tags: List[str] = Field(..., min_items=1)

validator = OutputValidator (client)

prompt = """Extract product info as JSON:
{
  "name": "product name",
  "price": price as number,
  "in_stock": true/false,
  "tags": ["tag1", "tag2"]
}

Product description: "Apple MacBook Pro M3, $1999, available now, laptop computer"

JSON:"""

product = validator.get_validated_output (prompt, Product)
print(f"Validated: {product.model_dump()}")
\`\`\`

## Handling Malformed Outputs

### Robust Parsing Strategies

\`\`\`python
import re
import json

class OutputParser:
    """
    Robust parsing of LLM outputs even when format isn't perfect.
    """
    
    @staticmethod
    def extract_json (text: str) -> dict:
        """
        Extract JSON from text even if surrounded by other content.
        """
        
        # Try direct JSON parse first
        try:
            return json.loads (text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON between code blocks
        code_block_pattern = r'\`\`\`(?: json) ?\\n?(.*?) \\n ? \`\`\`'
        matches = re.findall (code_block_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads (match)
            except json.JSONDecodeError:
                continue
        
        # Look for JSON-like structure
        json_pattern = r'\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}'
        matches = re.findall (json_pattern, text)
        
        for match in matches:
            try:
                return json.loads (match)
            except json.JSONDecodeError:
                continue
        
        raise ValueError("Could not extract valid JSON from text")
    
    @staticmethod
    def extract_code (text: str, language: str = None) -> str:
        """
        Extract code from markdown code blocks.
        """
        
        if language:
            pattern = f'\`\`\`{ language } \\n(.*?)\`\`\`'
        else:
            pattern = r'\`\`\`(?: \\w +) ?\\n(.*?)\`\`\`'
        
        matches = re.findall (pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return whole text
        return text.strip()
    
    @staticmethod
    def extract_list (text: str) -> List[str]:
        """
        Extract list items from various formats.
        """
        
        items = []
        
        # Try bullet points
        bullet_pattern = r'^[•\\-\\*]\\s+(.+)$'
        items.extend (re.findall (bullet_pattern, text, re.MULTILINE))
        
        # Try numbered lists
        number_pattern = r'^\\d+\\.\\s+(.+)$'
        items.extend (re.findall (number_pattern, text, re.MULTILINE))
        
        # If nothing found, split by newlines
        if not items:
            items = [line.strip() for line in text.split('\\n') if line.strip()]
        
        return items

# Usage examples
parser = OutputParser()

# Extract JSON from messy output
messy_output = """
Here\'s the data you requested:
\`\`\`json
{ "name": "Alice", "age": 30 }
\`\`\`
Hope this helps!
"""
data = parser.extract_json (messy_output)
print("Extracted JSON:", data)

# Extract code
code_output = """
Here's the implementation:
\`\`\`python
def hello():
print("Hello!")
    \`\`\`
"""
code = parser.extract_code (code_output, 'python')
print("Extracted code:", code)

# Extract list
list_output = """
Key points:
- Point one
- Point two  
- Point three
"""
items = parser.extract_list (list_output)
print("Extracted list:", items)
\`\`\`

## Retry Strategies for Format Violations

### Automatic Correction Loop

\`\`\`python
class FormatEnforcer:
    """
    Enforce output format with automatic retries.
    """
    
    def __init__(self, client, max_retries: int = 3):
        self.client = client
        self.max_retries = max_retries
    
    def enforce_format(
        self,
        task: str,
        format_spec: str,
        validation_func: callable,
        model: str = "gpt-4"
    ) -> str:
        """
        Get output in specific format, retry with feedback if wrong.
        """
        
        base_prompt = f"""{task}

{format_spec}"""
        
        prompt = base_prompt
        
        for attempt in range (self.max_retries):
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            output = response.choices[0].message.content
            
            # Validate format
            is_valid, error_msg = validation_func (output)
            
            if is_valid:
                print(f"✓ Valid format (attempt {attempt + 1})")
                return output
            
            print(f"✗ Invalid format (attempt {attempt + 1}): {error_msg}")
            
            if attempt < self.max_retries - 1:
                # Add corrective feedback
                prompt = f"""{base_prompt}

Previous attempt was invalid:
{output}

Error: {error_msg}

Please fix and provide output in the exact format specified."""
        
        raise ValueError (f"Could not get valid format after {self.max_retries} attempts")

# Example: Enforce CSV format
def validate_csv (output: str) -> tuple[bool, str]:
    """Validate CSV format."""
    
    lines = output.strip().split('\\n')
    
    if len (lines) < 2:
        return False, "CSV must have header and at least one data row"
    
    header_cols = len (lines[0].split(','))
    
    for i, line in enumerate (lines[1:], 2):
        cols = len (line.split(','))
        if cols != header_cols:
            return False, f"Line {i} has {cols} columns, expected {header_cols}"
    
    return True, ""

enforcer = FormatEnforcer (client)

csv_output = enforcer.enforce_format(
    task="Convert this data to CSV: Users: Alice (age 25), Bob (age 30)",
    format_spec="""Output as CSV with header row:
name,age
Alice,25
Bob,30

Rules:
- First row is headers
- Comma-separated
- No extra text""",
    validation_func=validate_csv
)

print("Valid CSV output:")
print(csv_output)
\`\`\`

## Production Checklist

✅ **Format Specification**
- Define exact output structure
- Provide examples
- Specify all required fields
- Document constraints
- Use schemas when possible

✅ **Validation**
- Validate all outputs
- Retry on format violations
- Parse robustly (handle edge cases)
- Provide error feedback to model
- Log validation failures

✅ **Type Safety**
- Use Pydantic models
- Use Instructor library
- Validate types at runtime
- Catch errors early
- Type hints everywhere

✅ **Error Handling**
- Graceful fallbacks
- Maximum retry limits
- Log failures for analysis
- Alert on repeated failures
- Manual review queue for edge cases

✅ **Testing**
- Test parsing logic thoroughly
- Test with malformed outputs
- Test edge cases
- Validate against schemas
- Monitor production parsing errors

## Key Takeaways

1. **JSON is the gold standard** - Easiest to parse and validate
2. **Use Instructor for type safety** - Pydantic + LLM = reliable outputs
3. **Always validate outputs** - Never trust LLM to get format perfect
4. **Retry with feedback** - Tell model what went wrong
5. **Parse robustly** - Handle variations and errors gracefully
6. **Schema enforcement works** - Significantly improves consistency
7. **XML/Markdown alternatives** - Sometimes better for hierarchical/mixed content
8. **Type safety prevents bugs** - Catch errors at parse time, not runtime
9. **Production needs validation** - Testing and monitoring essential
10. **Format control = reliability** - Predictable outputs enable automation

## Next Steps

Now that you understand output format control, you're ready to explore **Context Management & Truncation** - learning how to handle long documents and manage context windows effectively.`,
};
