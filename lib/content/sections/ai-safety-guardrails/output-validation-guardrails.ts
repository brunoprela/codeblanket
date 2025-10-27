export const outputValidationGuardrailsSection = `
# Output Validation & Guardrails

## Introduction

Output validation ensures LLM responses meet quality standards, follow schemas, and don't contain unwanted content. Guardrails are programmatic constraints that enforce these requirements automatically.

This section covers schema validation, quality checks, the Guardrails library, and building comprehensive output validation systems.

## Why Output Validation Matters

### Risks of Unvalidated Outputs

1. **Malformed Data**: Invalid JSON, broken code, incorrect formats
2. **Quality Issues**: Incomplete responses, off-topic content, poor quality
3. **Security Risks**: Injected code, malicious content, PII leaks
4. **Business Logic Violations**: Responses that don't meet requirements
5. **User Experience**: Confusing, unhelpful, or inappropriate responses

### Production Requirements

Output validation must:
- **Enforce schemas**: Ensure structured data matches expected format
- **Check quality**: Validate response completeness and relevance
- **Apply constraints**: Enforce business rules and requirements
- **Fail gracefully**: Handle validation failures with retries or fallbacks
- **Be fast**: Add minimal latency to response time
- **Be configurable**: Different validation rules for different use cases

## Schema Validation

### JSON Schema Validation

\`\`\`python
import json
from jsonschema import validate, ValidationError
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of schema validation"""
    is_valid: bool
    data: Any
    errors: list
    warnings: list

class JSONSchemaValidator:
    """Validate LLM outputs against JSON schemas"""

    def __init__(self):
        # Define schemas for different output types
        self.schemas = {
            'user_profile': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer', 'minimum': 0, 'maximum': 150},
                    'email': {'type': 'string', 'format': 'email'},
                    'interests': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'minItems': 1
                    }
                },
                'required': ['name', 'email']
            },
            'code_snippet': {
                'type': 'object',
                'properties': {
                    'language': {'type': 'string', 'enum': ['python', 'javascript', 'java']},
                    'code': {'type': 'string', 'minLength': 1},
                    'explanation': {'type': 'string'},
                    'test_cases': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'input': {'type': 'string'},
                                'expected_output': {'type': 'string'}
                            },
                            'required': ['input', 'expected_output']
                        }
                    }
                },
                'required': ['language', 'code']
            }
        }

    def validate_output(
        self,
        output: str,
        schema_name: str
    ) -> ValidationResult:
        """
        Validate LLM JSON output against schema.

        Args:
            output: LLM output string (should be JSON)
            schema_name: Name of schema to validate against
        """

        errors = []
        warnings = []

        # Get schema
        schema = self.schemas.get (schema_name)
        if not schema:
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=[f"Schema '{schema_name}' not found"],
                warnings=[]
            )

        # Parse JSON
        try:
            data = json.loads (output)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=[f"Invalid JSON: {str (e)}"],
                warnings=[]
            )

        # Validate against schema
        try:
            validate (instance=data, schema=schema)
            return ValidationResult(
                is_valid=True,
                data=data,
                errors=[],
                warnings=warnings
            )
        except ValidationError as e:
            errors.append (f"Schema validation error: {e.message}")
            return ValidationResult(
                is_valid=False,
                data=data,
                errors=errors,
                warnings=warnings
            )

# Example usage
validator = JSONSchemaValidator()

# Valid output
valid_output = json.dumps({
    'name': 'John Doe',
    'age': 30,
    'email': 'john@example.com',
    'interests': ['coding', 'reading']
})

result = validator.validate_output (valid_output, 'user_profile')
print(f"Valid: {result.is_valid}")
if not result.is_valid:
    print(f"Errors: {result.errors}")

# Invalid output (missing required field)
invalid_output = json.dumps({
    'name': 'John Doe',
    'age': 30
    # Missing required 'email'
})

result = validator.validate_output (invalid_output, 'user_profile')
print(f"\\nValid: {result.is_valid}")
print(f"Errors: {result.errors}")
\`\`\`

### Pydantic Models

\`\`\`python
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional
from enum import Enum

class ProgrammingLanguage (str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    TYPESCRIPT = "typescript"

class TestCase(BaseModel):
    """Test case for code"""
    input: str = Field(..., min_length=1)
    expected_output: str = Field(..., min_length=1)
    description: Optional[str] = None

class CodeSnippet(BaseModel):
    """Validated code snippet model"""
    language: ProgrammingLanguage
    code: str = Field(..., min_length=10, max_length=10000)
    explanation: str = Field(..., min_length=10)
    test_cases: List[TestCase] = Field (default_factory=list)

    @validator('code')
    def validate_code (cls, v, values):
        """Custom validation for code"""
        language = values.get('language')

        # Language-specific validation
        if language == ProgrammingLanguage.PYTHON:
            # Check for basic Python syntax
            if 'def ' not in v and 'class ' not in v:
                raise ValueError("Python code should contain function or class definition")

        # Check for potentially dangerous code
        dangerous_patterns = ['eval(', 'exec(', '__import__']
        if any (pattern in v for pattern in dangerous_patterns):
            raise ValueError("Code contains potentially dangerous patterns")

        return v

class Pydantic Validator:
    """Validate LLM outputs using Pydantic models"""

    def __init__(self):
        self.models = {
            'code_snippet': CodeSnippet,
            # Add more models as needed
        }

    def validate_output(
        self,
        output: str,
        model_name: str
    ) -> ValidationResult:
        """Validate output using Pydantic model"""

        # Get model
        model_class = self.models.get (model_name)
        if not model_class:
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=[f"Model '{model_name}' not found"],
                warnings=[]
            )

        # Parse JSON
        try:
            data = json.loads (output)
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=[f"Invalid JSON: {str (e)}"],
                warnings=[]
            )

        # Validate using Pydantic
        try:
            validated_data = model_class(**data)
            return ValidationResult(
                is_valid=True,
                data=validated_data.dict(),
                errors=[],
                warnings=[]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                data=data,
                errors=[str (e)],
                warnings=[]
            )

# Example usage
validator = PydanticValidator()

output = json.dumps({
    'language': 'python',
    'code': 'def hello():\\n    print("Hello, world!")',
    'explanation': 'A simple hello world function',
    'test_cases': [
        {'input': 'hello()', 'expected_output': 'Hello, world!'}
    ]
})

result = validator.validate_output (output, 'code_snippet')
print(f"Valid: {result.is_valid}")
if result.is_valid:
    print(f"Data: {result.data}")
\`\`\`

## Guardrails Library

\`\`\`python
# Using the Guardrails AI library for validation
import guardrails as gd
from guardrails.validators import ValidLength, ValidRange, ValidChoices

# Define a Guard with validators
guard = gd.Guard.from_rail_string("""
<rail version="0.1">
<output>
    <object name="user_profile">
        <string name="name" validators="valid-length: 2 50" />
        <integer name="age" validators="valid-range: 0 150" />
        <string name="email" format="email" />
        <list name="interests">
            <string validators="valid-length: 1 50" />
        </list>
        <string name="bio" validators="valid-length: 10 500" />
    </object>
</output>
</rail>
""")

class GuardrailsValidator:
    """Validate using Guardrails AI library"""

    def __init__(self):
        self.guards = {}
        self._setup_guards()

    def _setup_guards (self):
        """Setup predefined guards"""

        # Guard for code output
        self.guards['code'] = gd.Guard.from_rail_string("""
<rail version="0.1">
<output>
    <object name="code_output">
        <string name="language" validators="valid-choices: {['python', 'javascript', 'java']}" />
        <string name="code" validators="valid-length: 10 10000" />
        <string name="explanation" validators="valid-length: 10 1000" />
    </object>
</output>
</rail>
""")

        # Guard for summaries
        self.guards['summary'] = gd.Guard.from_rail_string("""
<rail version="0.1">
<output>
    <string name="summary" validators="valid-length: 50 500" />
</output>
</rail>
""")

    def validate_with_guard(
        self,
        output: str,
        guard_name: str
    ) -> Dict:
        """Validate output using named guard"""

        guard = self.guards.get (guard_name)
        if not guard:
            return {
                'valid': False,
                'error': f"Guard '{guard_name}' not found"
            }

        try:
            validated_output = guard.parse (output)
            return {
                'valid': True,
                'validated_output': validated_output,
                'validation_passed': True
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str (e),
                'original_output': output
            }

    def validate_and_fix(
        self,
        llm_function,
        guard_name: str,
        max_retries: int = 3
    ) -> Dict:
        """
        Validate LLM output and automatically retry/fix if invalid.

        Args:
            llm_function: Function that calls LLM
            guard_name: Name of guard to use
            max_retries: Maximum retry attempts
        """

        guard = self.guards.get (guard_name)
        if not guard:
            return {'valid': False, 'error': 'Guard not found'}

        for attempt in range (max_retries):
            try:
                # Call LLM
                output = llm_function()

                # Validate
                validated = guard.parse (output)

                return {
                    'valid': True,
                    'validated_output': validated,
                    'attempts': attempt + 1
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        'valid': False,
                        'error': str (e),
                        'attempts': attempt + 1
                    }
                continue

        return {'valid': False, 'error': 'Max retries exceeded'}

# Example usage
validator = GuardrailsValidator()

code_output = """
{
    "language": "python",
    "code": "def add (a, b):\\n    return a + b",
    "explanation": "A simple addition function"
}
"""

result = validator.validate_with_guard (code_output, 'code')
print(f"Valid: {result['valid']}")
\`\`\`

## Quality Validation

\`\`\`python
class QualityValidator:
    """Validate output quality"""

    def __init__(self):
        self.min_length = 50
        self.max_length = 5000

    def validate_quality(
        self,
        output: str,
        prompt: str
    ) -> Dict:
        """
        Validate output quality.

        Checks:
        1. Length (not too short or long)
        2. Relevance to prompt
        3. Completeness
        4. Coherence
        """

        issues = []
        warnings = []

        # Check 1: Length
        if len (output) < self.min_length:
            issues.append (f"Output too short: {len (output)} chars (min: {self.min_length})")
        elif len (output) > self.max_length:
            warnings.append (f"Output very long: {len (output)} chars (max: {self.max_length})")

        # Check 2: Completeness
        if output.endswith('...') or 'continued' in output.lower():
            issues.append("Output appears incomplete")

        # Check 3: Relevance (simple keyword matching)
        relevance_score = self._check_relevance (prompt, output)
        if relevance_score < 0.3:
            issues.append (f"Low relevance score: {relevance_score:.2f}")

        # Check 4: Coherence (simple checks)
        coherence_issues = self._check_coherence (output)
        issues.extend (coherence_issues)

        is_valid = len (issues) == 0

        return {
            'is_valid': is_valid,
            'quality_score': self._calculate_quality_score (issues, warnings),
            'issues': issues,
            'warnings': warnings,
            'relevance_score': relevance_score
        }

    def _check_relevance (self, prompt: str, output: str) -> float:
        """Check if output is relevant to prompt"""

        # Extract keywords from prompt
        prompt_words = set (prompt.lower().split())
        output_words = set (output.lower().split())

        # Calculate overlap
        overlap = len (prompt_words.intersection (output_words))
        relevance = overlap / max (len (prompt_words), 1)

        return relevance

    def _check_coherence (self, output: str) -> List[str]:
        """Check output coherence"""

        issues = []

        # Check for repeated phrases
        sentences = output.split('.')
        if len (sentences) > len (set (sentences)):
            issues.append("Contains repeated sentences")

        # Check for gibberish (very high ratio of non-words)
        words = output.split()
        if words:
            # Simple check: words should be mostly alphabetic
            alphabetic_ratio = sum(1 for w in words if w.isalpha()) / len (words)
            if alphabetic_ratio < 0.7:
                issues.append("High ratio of non-standard characters")

        return issues

    def _calculate_quality_score(
        self,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate overall quality score"""

        score = 1.0
        score -= len (issues) * 0.2
        score -= len (warnings) * 0.1

        return max(0.0, min(1.0, score))

# Example usage
validator = QualityValidator()
prompt = "Explain how photosynthesis works"
output = "Photosynthesis is the process by which plants convert sunlight into energy through a complex series of chemical reactions involving chlorophyll, carbon dioxide, and water."

result = validator.validate_quality (output, prompt)
print(f"Valid: {result['is_valid']}")
print(f"Quality score: {result['quality_score']:.2f}")
print(f"Issues: {result['issues']}")
\`\`\`

## Comprehensive Output Validation System

\`\`\`python
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: ValidationSeverity
    message: str
    validator: str

@dataclass
class ComprehensiveValidationResult:
    """Complete validation result"""
    is_valid: bool
    validated_data: Optional[Any]
    issues: List[ValidationIssue]
    quality_score: float
    should_retry: bool
    should_use_fallback: bool

class ComprehensiveValidator:
    """
    Comprehensive output validation system combining:
    1. Schema validation
    2. Quality validation
    3. Safety checks
    4. Business logic validation
    """

    def __init__(self):
        self.schema_validator = JSONSchemaValidator()
        self.quality_validator = QualityValidator()
        self.pii_detector = RegexPIIDetector()  # From previous section

    def validate(
        self,
        output: str,
        prompt: str,
        schema_name: Optional[str] = None,
        custom_validators: Optional[List[Callable]] = None
    ) -> ComprehensiveValidationResult:
        """
        Comprehensive validation of LLM output.

        Args:
            output: LLM output to validate
            prompt: Original prompt
            schema_name: Optional schema to validate against
            custom_validators: Optional custom validation functions
        """

        issues = []

        # Step 1: Schema validation (if schema provided)
        validated_data = None
        if schema_name:
            schema_result = self.schema_validator.validate_output (output, schema_name)
            if not schema_result.is_valid:
                for error in schema_result.errors:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=error,
                        validator='schema'
                    ))
            validated_data = schema_result.data

        # Step 2: Quality validation
        quality_result = self.quality_validator.validate_quality (output, prompt)
        for issue in quality_result['issues']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=issue,
                validator='quality'
            ))
        for warning in quality_result['warnings']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=warning,
                validator='quality'
            ))

        # Step 3: Safety checks (PII detection)
        pii_matches = self.pii_detector.detect (output)
        if pii_matches:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"PII detected: {len (pii_matches)} instances",
                validator='safety'
            ))

        # Step 4: Custom validators
        if custom_validators:
            for validator_func in custom_validators:
                try:
                    custom_result = validator_func (output)
                    if not custom_result.get('valid', True):
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=custom_result.get('message', 'Custom validation failed'),
                            validator='custom'
                        ))
                except Exception as e:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Custom validator error: {str (e)}",
                        validator='custom'
                    ))

        # Make final decision
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len (errors) == 0

        # Determine action
        should_retry = len (errors) > 0 and quality_result['quality_score'] > 0.3
        should_use_fallback = len (errors) > 0 and not should_retry

        return ComprehensiveValidationResult(
            is_valid=is_valid,
            validated_data=validated_data,
            issues=issues,
            quality_score=quality_result['quality_score'],
            should_retry=should_retry,
            should_use_fallback=should_use_fallback
        )

# Example usage
validator = ComprehensiveValidator()

# Custom validator example
def custom_no_urls (output: str) -> Dict:
    """Custom validator: no URLs allowed"""
    import re
    urls = re.findall (r'https?://\\S+', output)
    return {
        'valid': len (urls) == 0,
        'message': f"Found {len (urls)} URLs" if urls else "OK"
    }

output = json.dumps({
    'name': 'John Doe',
    'age': 30,
    'email': 'john@example.com',
    'interests': ['coding']
})

result = validator.validate(
    output=output,
    prompt="Generate a user profile",
    schema_name='user_profile',
    custom_validators=[custom_no_urls]
)

print(f"Valid: {result.is_valid}")
print(f"Quality score: {result.quality_score:.2f}")
print(f"Should retry: {result.should_retry}")
for issue in result.issues:
    print(f"  [{issue.severity.value}] {issue.validator}: {issue.message}")
\`\`\`

## Key Takeaways

1. **Schema validation**: Use JSON Schema or Pydantic for structure
2. **Quality checks**: Validate length, relevance, completeness
3. **Safety validation**: Check for PII, injection, harmful content
4. **Guardrails library**: Powerful tool for validation with retries
5. **Custom validators**: Add domain-specific validation
6. **Graceful failures**: Retry or use fallbacks when validation fails
7. **Monitor validation rates**: Track how often outputs fail validation

## Production Checklist

- [ ] Schema validation for structured outputs
- [ ] Quality validation (length, relevance, completeness)
- [ ] Safety checks (PII, injection, harmful content)
- [ ] Custom validators for business logic
- [ ] Retry logic for failed validations
- [ ] Fallback responses for repeated failures
- [ ] Validation metrics and monitoring
- [ ] A/B testing of validation rules
- [ ] Documentation of validation requirements
- [ ] Regular review of validation failures

Output validation is your last line of defense—ensure nothing problematic reaches users.
`;
