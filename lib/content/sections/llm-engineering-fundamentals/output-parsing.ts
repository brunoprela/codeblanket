/**
 * Output Parsing & Structured Data Section
 * Module 1: LLM Engineering Fundamentals
 */

export const outputparsingSection = {
  id: 'output-parsing',
  title: 'Output Parsing & Structured Data',
  content: `# Output Parsing & Structured Data

Master extracting structured data from LLM outputs for reliable production applications.

## The Output Parsing Challenge

LLMs return text. Your application needs structured data.

### The Problem

\`\`\`python
"""
You ask LLM:
"Extract name and email from: John Doe (john@email.com)"

LLM might return:
- "The name is John Doe and the email is john@email.com"
- "Name: John Doe, Email: john@email.com"
- "John Doe, john@email.com"
- "name='John Doe' email='john@email.com'"

You need:
{
    "name": "John Doe",
    "email": "john@email.com"
}

HOW DO YOU PARSE THIS RELIABLY?
"""
\`\`\`

## Strategy 1: JSON Mode

Modern LLMs support JSON mode for structured outputs.

### OpenAI JSON Mode

\`\`\`python
from openai import OpenAI
import json

client = OpenAI()

def extract_person_json (text: str) -> dict:
    """
    Extract person info using JSON mode.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",  # Requires model that supports JSON mode
        messages=[
            {
                "role": "system",
                "content": "Extract person information as JSON with fields: name, email, age"
            },
            {
                "role": "user",
                "content": f"Extract from: {text}"
            }
        ],
        response_format={"type": "json_object"}  # ← Enable JSON mode!
    )

    # Parse JSON
    json_text = response.choices[0].message.content
    return json.loads (json_text)

# Usage
text = "Hi, I'm Alice (alice@email.com) and I'm 25 years old"

result = extract_person_json (text)
print(type (result))  # <class 'dict'>
print(result)  # {'name': 'Alice', 'email': 'alice@email.com', 'age': 25}

# Can use directly in code!
print(f"Name: {result['name']}")
print(f"Email: {result['email']}")
\`\`\`

### JSON Mode with Schema

\`\`\`python
import json
from openai import OpenAI

client = OpenAI()

def extract_with_schema (text: str, schema: dict) -> dict:
    """
    Extract data with specific JSON schema.
    """

    schema_str = json.dumps (schema, indent=2)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": f"""Extract information as JSON matching this schema:

{schema_str}

Return ONLY valid JSON matching this schema."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads (response.choices[0].message.content)

# Define schema
person_schema = {
    "name": "string",
    "email": "string",
    "age": "number",
    "interests": ["string"]
}

# Extract
text = "John Smith, john.smith@example.com, age 30, loves Python and AI"

result = extract_with_schema (text, person_schema)
print(json.dumps (result, indent=2))
\`\`\`

## Strategy 2: Pydantic Models

Use Pydantic for type-safe parsing with validation.

### Basic Pydantic Parsing

\`\`\`python
# pip install pydantic

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from openai import OpenAI
import json

class Person(BaseModel):
    """Person data model with validation."""
    name: str
    email: EmailStr  # Validates email format!
    age: Optional[int] = None
    interests: List[str] = Field (default_factory=list)

def extract_person_typed (text: str) -> Person:
    """
    Extract person with Pydantic validation.
    """
    client = OpenAI()

    # Get schema from Pydantic model
    schema = Person.schema()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": f"""Extract person information as JSON matching this schema:

{json.dumps (schema, indent=2)}

Return ONLY valid JSON."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format={"type": "json_object"}
    )

    # Parse and validate
    json_data = json.loads (response.choices[0].message.content)

    # Pydantic validates automatically!
    return Person(**json_data)

# Usage
text = "Alice Johnson (alice.j@company.com), 28 years old, interested in ML and NLP"

person = extract_person_typed (text)
print(person)  # Person (name='Alice Johnson', email='alice.j@company.com', ...)
print(type (person))  # <class 'Person'>

# Type-safe access
print(f"Email domain: {person.email.split('@')[1]}")

# Validation works!
try:
    bad_person = Person (name="Bob", email="invalid-email")  # Will fail!
except Exception as e:
    print(f"Validation error: {e}")
\`\`\`

### Complex Pydantic Models

\`\`\`python
from pydantic import BaseModel, validator
from typing import List, Literal
from datetime import date

class Address(BaseModel):
    """Nested address model."""
    street: str
    city: str
    state: str
    zip_code: str

class Employee(BaseModel):
    """Complex employee model."""
    name: str
    employee_id: str
    email: EmailStr
    department: Literal["Engineering", "Sales", "Marketing", "HR"]
    start_date: date
    address: Address
    skills: List[str]

    @validator('employee_id')
    def validate_employee_id (cls, v):
        """Custom validation for employee ID."""
        if not v.startswith('EMP'):
            raise ValueError('Employee ID must start with EMP')
        return v

# Extract function
def extract_employee (text: str) -> Employee:
    """Extract employee with nested models and validation."""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": f"""Extract employee information as JSON.

Schema:
{json.dumps(Employee.schema(), indent=2)}

Return ONLY valid JSON."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format={"type": "json_object"}
    )

    json_data = json.loads (response.choices[0].message.content)
    return Employee(**json_data)

# Test
text = """
Employee: Sarah Miller
ID: EMP12345
Email: sarah.m@company.com
Department: Engineering
Started: 2023-01-15
Address: 123 Main St, San Francisco, CA 94102
Skills: Python, Machine Learning, System Design
"""

employee = extract_employee (text)
print(f"Name: {employee.name}")
print(f"Department: {employee.department}")
print(f"City: {employee.address.city}")
print(f"Skills: {', '.join (employee.skills)}")
\`\`\`

## Strategy 3: The Instructor Library

Instructor simplifies structured outputs from LLMs.

### Basic Instructor Usage

\`\`\`python
# pip install instructor

import instructor
from openai import OpenAI
from pydantic import BaseModel

# Patch OpenAI client
client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    age: int
    email: str

# Extract with one line!
user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=User,  # ← Magic happens here!
    messages=[
        {"role": "user", "content": "Extract: John Doe, 30 years old, john@email.com"}
    ]
)

print(user)  # User (name='John Doe', age=30, email='john@email.com')
print(type (user))  # <class 'User'>
\`\`\`

### Instructor with Validation

\`\`\`python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, validator
from typing import List

client = instructor.from_openai(OpenAI())

class Recipe(BaseModel):
    """Recipe with validation."""
    name: str = Field (description="Name of the dish")
    ingredients: List[str] = Field (description="List of ingredients")
    instructions: List[str] = Field (description="Step-by-step instructions")
    prep_time_minutes: int = Field (ge=0, description="Prep time in minutes")

    @validator('ingredients')
    def validate_ingredients (cls, v):
        if len (v) < 2:
            raise ValueError("Recipe must have at least 2 ingredients")
        return v

# Extract recipe
recipe = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=Recipe,
    messages=[
        {"role": "user", "content": "Give me a simple pasta recipe"}
    ]
)

print(f"Recipe: {recipe.name}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Ingredients ({len (recipe.ingredients)}): {recipe.ingredients}")
\`\`\`

### Instructor with Lists

\`\`\`python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List

client = instructor.from_openai(OpenAI())

class Person(BaseModel):
    name: str
    occupation: str

class People(BaseModel):
    """Collection of people."""
    people: List[Person]

# Extract multiple items
result = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=People,
    messages=[
        {"role": "user", "content": """
        Extract all people mentioned:

        John is a software engineer.
        Sarah works as a doctor.
        Mike is a teacher.
        """}
    ]
)

print(f"Found {len (result.people)} people:")
for person in result.people:
    print(f"  - {person.name}: {person.occupation}")
\`\`\`

## Strategy 4: Regex Parsing

When JSON mode isn't available, use regex.

### Basic Regex Extraction

\`\`\`python
import re
from typing import Optional

def extract_email_regex (text: str) -> Optional[str]:
    """Extract email using regex."""
    pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
    match = re.search (pattern, text)
    return match.group(0) if match else None

def extract_phone_regex (text: str) -> Optional[str]:
    """Extract phone number using regex."""
    # Matches formats: (123) 456-7890, 123-456-7890, 1234567890
    pattern = r'\\(?(\\d{3})\\)?[-. ]?(\\d{3})[-. ]?(\\d{4})'
    match = re.search (pattern, text)
    return match.group(0) if match else None

# Usage
text = "Contact John at john@email.com or (555) 123-4567"

email = extract_email_regex (text)
phone = extract_phone_regex (text)

print(f"Email: {email}")
print(f"Phone: {phone}")
\`\`\`

### Structured Regex Parsing

\`\`\`python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class Contact:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

def parse_contact_regex (llm_output: str) -> Contact:
    """
    Parse contact from LLM output using regex.
    Handles various output formats.
    """

    # Try to extract name
    name_match = re.search (r'Name:?\\s*([^\\n,]+)', llm_output, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else None

    # Extract email
    email_match = re.search(
        r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
        llm_output
    )
    email = email_match.group(0) if email_match else None

    # Extract phone
    phone_match = re.search(
        r'\\(?(\\d{3})\\)?[-. ]?(\\d{3})[-. ]?(\\d{4})',
        llm_output
    )
    phone = phone_match.group(0) if phone_match else None

    return Contact (name=name, email=email, phone=phone)

# Test with different LLM output formats
outputs = [
    "Name: Alice, Email: alice@test.com, Phone: (555) 123-4567",
    "The person is Bob (bob@example.com), phone number 5551234567",
    "Contact: Charlie | charlie@mail.com | 555-123-4567"
]

for output in outputs:
    contact = parse_contact_regex (output)
    print(f"Parsed: {contact}")
\`\`\`

## Handling Parse Failures

Always handle cases where parsing fails.

### Robust Parser with Fallback

\`\`\`python
from typing import Optional, Union
from pydantic import BaseModel, ValidationError
import json

class Person(BaseModel):
    name: str
    email: str
    age: Optional[int] = None

def parse_with_retry(
    llm_output: str,
    model_class: type[BaseModel],
    max_retries: int = 3
) -> Union[BaseModel, dict]:
    """
    Try to parse LLM output, retry if it fails.
    """

    # Try 1: Direct JSON parsing
    try:
        data = json.loads (llm_output)
        return model_class(**data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Try 2: Extract JSON from markdown
    try:
        # Look for \`\`\`json ... \`\`\` blocks
        json_match = re.search (r'\`\`\`json\\s*({.*?}) \\s* \`\`\`', llm_output, re.DOTALL)
        if json_match:
            data = json.loads (json_match.group(1))
            return model_class(**data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Try 3: Extract any {...} block
    try:
        json_match = re.search (r'{.*}', llm_output, re.DOTALL)
        if json_match:
            data = json.loads (json_match.group(0))
            return model_class(**data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # Failed - return raw dict or None
    return {'raw_output': llm_output, 'parse_failed': True}

# Test with various formats
outputs = [
    '{"name": "Alice", "email": "alice@test.com", "age": 25}',  # Valid JSON
    '\`\`\`json\\n{ "name": "Bob", "email": "bob@test.com" } \\n\`\`\`',  # Markdown
    'The person is {"name": "Charlie", "email": "charlie@test.com"}',  # Embedded
    'Name: Dave, Email: dave@test.com'  # Will fail
]

for output in outputs:
    result = parse_with_retry (output, Person)
    print(f"Input: {output[:50]}...")
    print(f"Result: {result}")
    print()
\`\`\`

### Retry with LLM

If parsing fails, ask LLM to fix it.

\`\`\`python
from openai import OpenAI
import json

client = OpenAI()

def extract_with_retry(
    text: str,
    schema: dict,
    max_retries: int = 3
) -> dict:
    """
    Extract data, retry with feedback if parsing fails.
    """

    messages = [
        {
            "role": "system",
            "content": f"""Extract data as valid JSON matching this schema:
{json.dumps (schema, indent=2)}

Return ONLY valid JSON, no extra text."""
        },
        {
            "role": "user",
            "content": f"Extract from: {text}"
        }
    ]

    for attempt in range (max_retries):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            response_format={"type": "json_object"}
        )

        output = response.choices[0].message.content

        # Try to parse
        try:
            data = json.loads (output)

            # Validate against schema (simple check)
            if all (key in data for key in schema.keys()):
                return data
            else:
                # Missing fields - ask for correction
                missing = [k for k in schema.keys() if k not in data]
                messages.append({
                    "role": "assistant",
                    "content": output
                })
                messages.append({
                    "role": "user",
                    "content": f"Missing fields: {missing}. Please include them."
                })

        except json.JSONDecodeError as e:
            # Invalid JSON - ask for correction
            messages.append({
                "role": "assistant",
                "content": output
            })
            messages.append({
                "role": "user",
                "content": f"Invalid JSON: {e}. Please provide valid JSON."
            })

    raise ValueError (f"Failed to parse after {max_retries} attempts")

# Usage
schema = {"name": "string", "email": "string", "age": "number"}
text = "Alice Johnson, alice@email.com, 25 years old"

result = extract_with_retry (text, schema)
print(json.dumps (result, indent=2))
\`\`\`

## Production Parsing System

\`\`\`python
from typing import TypeVar, Type, Union, Optional
from pydantic import BaseModel, ValidationError
import instructor
from openai import OpenAI
import json

T = TypeVar('T', bound=BaseModel)

class ProductionParser:
    """
    Production-ready parser with multiple strategies.
    """

    def __init__(self, use_instructor: bool = True):
        if use_instructor:
            self.client = instructor.from_openai(OpenAI())
        else:
            self.client = OpenAI()
        self.use_instructor = use_instructor

    def extract(
        self,
        text: str,
        model_class: Type[T],
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Union[T, None]:
        """
        Extract structured data with automatic retries.
        """

        if self.use_instructor:
            return self._extract_with_instructor (text, model_class, system_prompt)
        else:
            return self._extract_with_json_mode (text, model_class, system_prompt, max_retries)

    def _extract_with_instructor(
        self,
        text: str,
        model_class: Type[T],
        system_prompt: Optional[str]
    ) -> T:
        """Extract using instructor library."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})

        return self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=model_class,
            messages=messages
        )

    def _extract_with_json_mode(
        self,
        text: str,
        model_class: Type[T],
        system_prompt: Optional[str],
        max_retries: int
    ) -> Optional[T]:
        """Extract using JSON mode with retries."""

        schema = model_class.schema()

        messages = [
            {
                "role": "system",
                "content": system_prompt or f"Extract as JSON: {json.dumps (schema)}"
            },
            {
                "role": "user",
                "content": text
            }
        ]

        for attempt in range (max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=messages,
                    response_format={"type": "json_object"}
                )

                json_data = json.loads (response.choices[0].message.content)
                return model_class(**json_data)

            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < max_retries - 1:
                    # Add error feedback
                    messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Error: {e}. Please fix and return valid JSON."
                    })
                else:
                    return None

        return None

# Usage
from pydantic import EmailStr

class Contact(BaseModel):
    name: str
    email: EmailStr
    age: int

parser = ProductionParser (use_instructor=True)

text = "John Doe, 30 years old, contact at john.doe@email.com"

contact = parser.extract(
    text=text,
    model_class=Contact,
    system_prompt="Extract contact information accurately."
)

if contact:
    print(f"Parsed successfully: {contact}")
else:
    print("Failed to parse")
\`\`\`

## Key Takeaways

1. **Use JSON mode** when available - most reliable
2. **Pydantic validates** automatically - type safety!
3. **Instructor simplifies** structured extraction
4. **Always validate** parsed data
5. **Handle failures gracefully** - parsing can fail
6. **Retry with feedback** if parsing fails
7. **Define clear schemas** - helps LLM generate correctly
8. **Use temperature=0** for structured extraction
9. **Test edge cases** - empty fields, missing data
10. **Fallback to regex** when JSON mode unavailable

## Next Steps

Now you can extract structured data reliably. Next: **LLM Observability & Logging** - learning to monitor and debug LLM applications in production.`,
};
