export const piiDetectionRemovalSection = `
# PII Detection & Removal

## Introduction

Personally Identifiable Information (PII) leakage is one of the most serious risks in AI applications. A single exposed email, phone number, or social security number can lead to GDPR violations, privacy breaches, and significant legal consequences.

This section covers identifying PII, implementing detection systems, redacting sensitive information, and ensuring GDPR/CCPA compliance in production AI applications.

## Understanding PII

### What is PII?

Personally Identifiable Information is any data that could potentially identify a specific individual:

**Direct Identifiers**:
- Full name
- Email address
- Phone number
- Social Security Number (SSN)
- Driver's license number
- Passport number
- Credit card numbers
- Bank account numbers
- IP addresses
- Device IDs

**Indirect Identifiers** (can identify when combined):
- Date of birth
- Address
- ZIP code
- Gender + age + location
- Employment information
- Educational records
- Medical records

### Why PII Detection Matters

1. **Legal Compliance**: GDPR, CCPA, HIPAA require PII protection
2. **Privacy Protection**: Users' right to privacy
3. **Security**: PII is valuable to bad actors
4. **Trust**: Users need to trust your system
5. **Liability**: Breaches can cost millions in fines

### PII in AI Systems

PII can appear in:
- User prompts/inputs
- LLM outputs/completions
- Training data
- Logs and audit trails
- Error messages
- API responses

## Detection Strategies

### Regex Pattern Matching

\`\`\`python
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class PIIMatch:
    """Represents a PII match"""
    type: str
    value: str
    start: int
    end: int
    confidence: float

class RegexPIIDetector:
    """Detect PII using regex patterns"""

    def __init__(self):
        self.patterns = {
            'email': re.compile(
                r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
            ),
            'phone_us': re.compile(
                r'\\b(?:\\+?1[-.]?)?\\(?([0-9]{3})\\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\\b'
            ),
            'ssn': re.compile(
                r'\\b(?!000|666|9\\d{2})\\d{3}-(?!00)\\d{2}-(?!0{4})\\d{4}\\b'
            ),
            'credit_card': re.compile(
                r'\\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\\d{3})\\d{11})\\b'
            ),
            'ip_address': re.compile(
                r'\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b'
            ),
            'us_address': re.compile(
                r'\\b\\d{1,5}\\s+\\w+\\s+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\\b',
                re.IGNORECASE
            ),
            'zip_code': re.compile(
                r'\\b\\d{5}(?:-\\d{4})?\\b'
            ),
            'date_of_birth': re.compile(
                r'\\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\\d|3[01])/(?:19|20)\\d{2}\\b'
            ),
        }

    def detect (self, text: str) -> List[PIIMatch]:
        """Detect all PII in text"""

        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer (text):
                matches.append(PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=self._calculate_confidence (pii_type, match.group())
                ))

        # Sort by position
        matches.sort (key=lambda m: m.start)

        return matches

    def _calculate_confidence (self, pii_type: str, value: str) -> float:
        """Calculate confidence score for detected PII"""

        # Basic confidence scoring
        if pii_type == 'email':
            # Higher confidence for common domains
            common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
            return 0.95 if any (domain in value for domain in common_domains) else 0.85

        elif pii_type == 'credit_card':
            # Validate using Luhn algorithm
            return 0.99 if self._luhn_check (value.replace('-', ').replace(' ', ')) else 0.5

        elif pii_type == 'ssn':
            return 0.95  # Pattern is quite specific

        elif pii_type == 'phone_us':
            return 0.90

        elif pii_type == 'ip_address':
            # Check if valid IP range
            return 0.85 if self._is_valid_ip (value) else 0.5

        return 0.80  # Default confidence

    def _luhn_check (self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        def digits_of (n):
            return [int (d) for d in str (n)]

        digits = digits_of (card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum (odd_digits)
        for d in even_digits:
            checksum += sum (digits_of (d * 2))
        return checksum % 10 == 0

    def _is_valid_ip (self, ip: str) -> bool:
        """Check if IP address is valid"""
        parts = ip.split('.')
        return all(0 <= int (part) <= 255 for part in parts)

# Example usage
detector = RegexPIIDetector()
text = """
My email is john.doe@gmail.com and my phone is 555-123-4567.
My SSN is 123-45-6789 and credit card is 4532-1234-5678-9010.
"""

matches = detector.detect (text)
print(f"Found {len (matches)} PII matches:")
for match in matches:
    print(f"  - {match.type}: {match.value} (confidence: {match.confidence:.2f})")
\`\`\`

### Named Entity Recognition (NER)

\`\`\`python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict

class NERPIIDetector:
    """Detect PII using Named Entity Recognition"""

    def __init__(self):
        # Use a model fine-tuned for PII detection
        model_name = "StanfordAIMI/stanford-deidentifier-base"
        self.tokenizer = AutoTokenizer.from_pretrained (model_name)
        self.model = AutoModelForTokenClassification.from_pretrained (model_name)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def detect (self, text: str) -> List[Dict]:
        """Detect PII using NER"""

        try:
            # Run NER
            entities = self.ner_pipeline (text)

            # Filter for PII-related entities
            pii_entities = []
            pii_labels = {
                'PERSON', 'PER',           # Person names
                'EMAIL',                    # Email addresses
                'PHONE', 'PHONE_NUMBER',   # Phone numbers
                'SSN', 'ID',               # ID numbers
                'DATE', 'DOB',             # Dates (potential DOB)
                'ADDRESS', 'LOC',          # Addresses
                'ORG',                     # Organizations (employer)
                'CREDIT_CARD', 'CC',       # Credit cards
            }

            for entity in entities:
                if any (label in entity['entity_group'].upper() for label in pii_labels):
                    pii_entities.append({
                        'type': entity['entity_group'],
                        'value': entity['word'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score']
                    })

            return pii_entities

        except Exception as e:
            print(f"NER detection error: {e}")
            return []

# Example usage
ner_detector = NERPIIDetector()
text = "John Smith lives at 123 Main Street and works at Acme Corp."
entities = ner_detector.detect (text)

print(f"Found {len (entities)} PII entities:")
for entity in entities:
    print(f"  - {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
\`\`\`

### Multi-Method PII Detection

\`\`\`python
from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class PIIDetectionResult:
    """Complete PII detection result"""
    has_pii: bool
    pii_found: List[PIIMatch]
    pii_types: Set[str]
    confidence: float
    methods_used: List[str]

class ComprehensivePIIDetector:
    """
    Comprehensive PII detection using multiple methods:
    1. Regex patterns (fast, high precision)
    2. NER (slower, catches names and context)
    3. Custom rules (domain-specific)
    """

    def __init__(self):
        self.regex_detector = RegexPIIDetector()
        self.ner_detector = NERPIIDetector()

    def detect (self, text: str, use_ner: bool = True) -> PIIDetectionResult:
        """Detect PII using multiple methods"""

        all_pii = []
        methods_used = ['regex']

        # Method 1: Regex detection
        regex_matches = self.regex_detector.detect (text)
        all_pii.extend (regex_matches)

        # Method 2: NER detection (optional, slower)
        if use_ner:
            methods_used.append('ner')
            ner_entities = self.ner_detector.detect (text)

            # Convert NER entities to PIIMatch format
            for entity in ner_entities:
                # Avoid duplicates with regex matches
                if not self._is_duplicate (entity, all_pii):
                    all_pii.append(PIIMatch(
                        type=entity['type'],
                        value=entity['value'],
                        start=entity['start'],
                        end=entity['end'],
                        confidence=entity['confidence']
                    ))

        # Method 3: Custom rules
        methods_used.append('custom_rules')
        custom_pii = self._custom_detection (text)
        for item in custom_pii:
            if not self._is_duplicate (item, all_pii):
                all_pii.append (item)

        # Calculate overall confidence
        avg_confidence = (
            sum (pii.confidence for pii in all_pii) / len (all_pii)
            if all_pii else 0.0
        )

        return PIIDetectionResult(
            has_pii=len (all_pii) > 0,
            pii_found=all_pii,
            pii_types=set (pii.type for pii in all_pii),
            confidence=avg_confidence,
            methods_used=methods_used
        )

    def _is_duplicate (self, new_item: Dict, existing_items: List[PIIMatch]) -> bool:
        """Check if new item overlaps with existing detections"""
        new_start = new_item.get('start', 0)
        new_end = new_item.get('end', 0)

        for existing in existing_items:
            # Check for overlap
            if (new_start <= existing.end and new_end >= existing.start):
                return True

        return False

    def _custom_detection (self, text: str) -> List[PIIMatch]:
        """Custom detection rules for domain-specific PII"""

        custom_matches = []

        # Example: Detect passport numbers (simplified)
        import re
        passport_pattern = re.compile (r'\\b[A-Z]{2}\\d{7}\\b')
        for match in passport_pattern.finditer (text):
            custom_matches.append(PIIMatch(
                type='passport',
                value=match.group(),
                start=match.start(),
                end=match.end(),
                confidence=0.85
            ))

        # Add more custom rules as needed

        return custom_matches

# Example usage
detector = ComprehensivePIIDetector()
text = """
Hi, I'm Jane Doe. You can reach me at jane@example.com or 555-867-5309.
My address is 456 Oak Avenue, and my DOB is 03/15/1990.
"""

result = detector.detect (text)
print(f"PII detected: {result.has_pii}")
print(f"Types: {result.pii_types}")
print(f"Methods: {result.methods_used}")
print(f"\\nDetails:")
for pii in result.pii_found:
    print(f"  - {pii.type}: {pii.value} (confidence: {pii.confidence:.2f})")
\`\`\`

## Redaction Strategies

### Basic Redaction

\`\`\`python
class PIIRedactor:
    """Redact detected PII from text"""

    def __init__(self):
        self.detector = ComprehensivePIIDetector()

    def redact(
        self,
        text: str,
        redaction_char: str = '*',
        preserve_format: bool = True
    ) -> Tuple[str, PIIDetectionResult]:
        """
        Redact PII from text.

        Args:
            text: Input text
            redaction_char: Character to use for redaction
            preserve_format: Keep length/format of original

        Returns:
            (redacted_text, detection_result)
        """

        # Detect PII
        detection_result = self.detector.detect (text)

        if not detection_result.has_pii:
            return text, detection_result

        # Sort matches by position (reverse order to maintain indices)
        sorted_matches = sorted(
            detection_result.pii_found,
            key=lambda m: m.start,
            reverse=True
        )

        redacted = text
        for match in sorted_matches:
            if preserve_format:
                # Replace with same length of redaction chars
                replacement = redaction_char * len (match.value)
            else:
                # Replace with type label
                replacement = f"[{match.type.upper()}]"

            redacted = (
                redacted[:match.start] +
                replacement +
                redacted[match.end:]
            )

        return redacted, detection_result

# Example usage
redactor = PIIRedactor()
text = "Contact John Doe at john@example.com or 555-123-4567"
redacted, result = redactor.redact (text)

print(f"Original: {text}")
print(f"Redacted: {redacted}")
\`\`\`

### Selective Redaction

\`\`\`python
class SelectivePIIRedactor:
    """Redact only specific types of PII"""

    def __init__(self):
        self.detector = ComprehensivePIIDetector()

    def redact_selective(
        self,
        text: str,
        redact_types: Set[str],
        replacement_map: Dict[str, str] = None
    ) -> str:
        """
        Selectively redact certain PII types.

        Args:
            text: Input text
            redact_types: Set of PII types to redact (e.g., {'email', 'phone'})
            replacement_map: Custom replacements per type
        """

        # Default replacements
        if replacement_map is None:
            replacement_map = {
                'email': '[EMAIL_ADDRESS]',
                'phone_us': '[PHONE_NUMBER]',
                'ssn': '[SSN]',
                'credit_card': '[CREDIT_CARD]',
                'PERSON': '[NAME]',
                'ADDRESS': '[ADDRESS]',
            }

        # Detect PII
        detection_result = self.detector.detect (text)

        # Filter for specified types
        matches_to_redact = [
            m for m in detection_result.pii_found
            if m.type in redact_types
        ]

        # Sort in reverse order
        matches_to_redact.sort (key=lambda m: m.start, reverse=True)

        redacted = text
        for match in matches_to_redact:
            replacement = replacement_map.get (match.type, '[REDACTED]')
            redacted = (
                redacted[:match.start] +
                replacement +
                redacted[match.end:]
            )

        return redacted

# Example usage
redactor = SelectivePIIRedactor()
text = "John Doe (john@example.com) lives at 123 Main St. Call: 555-1234"

# Only redact email and phone
redacted = redactor.redact_selective(
    text,
    redact_types={'email', 'phone_us'}
)

print(f"Original: {text}")
print(f"Redacted: {redacted}")
# Output: "John Doe ([EMAIL_ADDRESS]) lives at 123 Main St. Call: [PHONE_NUMBER]"
\`\`\`

### Pseudonymization

\`\`\`python
import hashlib
from typing import Dict

class PIIPseudonymizer:
    """
    Pseudonymize PII instead of redacting completely.
    Allows data analysis while protecting privacy.
    """

    def __init__(self):
        self.detector = ComprehensivePIIDetector()
        self.pseudonym_map: Dict[str, str] = {}
        self.counter = 0

    def pseudonymize(
        self,
        text: str,
        consistent: bool = True
    ) -> Tuple[str, Dict]:
        """
        Pseudonymize PII in text.

        Args:
            text: Input text
            consistent: Use consistent pseudonyms for same values

        Returns:
            (pseudonymized_text, pseudonym_mapping)
        """

        detection_result = self.detector.detect (text)

        if not detection_result.has_pii:
            return text, {}

        # Sort matches by position (reverse)
        sorted_matches = sorted(
            detection_result.pii_found,
            key=lambda m: m.start,
            reverse=True
        )

        pseudonymized = text
        mapping = {}

        for match in sorted_matches:
            pseudonym = self._generate_pseudonym(
                match.type,
                match.value,
                consistent
            )

            mapping[match.value] = pseudonym
            pseudonymized = (
                pseudonymized[:match.start] +
                pseudonym +
                pseudonymized[match.end:]
            )

        return pseudonymized, mapping

    def _generate_pseudonym(
        self,
        pii_type: str,
        value: str,
        consistent: bool
    ) -> str:
        """Generate a pseudonym for a PII value"""

        if consistent and value in self.pseudonym_map:
            return self.pseudonym_map[value]

        # Generate pseudonym based on type
        if pii_type == 'email':
            pseudonym = f"user{self.counter}@example.com"
        elif pii_type == 'phone_us':
            pseudonym = f"555-{self.counter:04d}-{self.counter:04d}"
        elif pii_type in ['PERSON', 'PER']:
            pseudonym = f"Person_{self.counter}"
        elif pii_type == 'ADDRESS':
            pseudonym = f"{self.counter} Example Street"
        elif pii_type == 'ssn':
            pseudonym = f"***-**-{self.counter:04d}"
        else:
            pseudonym = f"[{pii_type}_{self.counter}]"

        self.counter += 1

        if consistent:
            self.pseudonym_map[value] = pseudonym

        return pseudonym

# Example usage
pseudonymizer = PIIPseudonymizer()
text = """
Jane Doe (jane@company.com) and John Smith (jane@company.com)
called from 555-1234 and 555-5678.
"""

pseudonymized, mapping = pseudonymizer.pseudonymize (text, consistent=True)

print(f"Original: {text}")
print(f"Pseudonymized: {pseudonymized}")
print(f"\\nMapping:")
for original, pseudo in mapping.items():
    print(f"  {original} -> {pseudo}")
\`\`\`

## GDPR Compliance

\`\`\`python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class GDPRCompliantPIIHandler:
    """
    GDPR-compliant PII handling system.

    GDPR Requirements:
    1. Right to access (Article 15)
    2. Right to rectification (Article 16)
    3. Right to erasure/"right to be forgotten" (Article 17)
    4. Right to data portability (Article 20)
    5. Data minimization (Article 5)
    6. Storage limitation (Article 5)
    """

    def __init__(self):
        self.detector = ComprehensivePIIDetector()
        self.redactor = PIIRedactor()
        self.data_registry: Dict[str, Dict] = {}  # Use database in production

    def process_user_data(
        self,
        user_id: str,
        data: str,
        purpose: str,
        retention_days: int = 30
    ) -> Dict:
        """
        Process user data in GDPR-compliant manner.

        Args:
            user_id: Unique user identifier
            data: Data containing potential PII
            purpose: Purpose of data processing (required by GDPR)
            retention_days: How long to retain data
        """

        # Detect PII
        detection_result = self.detector.detect (data)

        # Create data processing record
        record = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'purpose': purpose,
            'pii_detected': detection_result.has_pii,
            'pii_types': list (detection_result.pii_types),
            'retention_until': (
                datetime.now() + timedelta (days=retention_days)
            ).isoformat(),
            'data_hash': self._hash_data (data),  # Store hash, not raw data
            'consent_obtained': True,  # Track consent
        }

        # Store record
        self.data_registry[user_id] = record

        return {
            'processed': True,
            'pii_detected': detection_result.has_pii,
            'pii_types': list (detection_result.pii_types),
            'retention_period': retention_days,
            'record_id': user_id
        }

    def export_user_data (self, user_id: str) -> Dict:
        """
        Export all data for a user (GDPR Article 20 - Data Portability).
        """
        if user_id not in self.data_registry:
            return {'error': 'User not found'}

        record = self.data_registry[user_id]

        return {
            'user_id': user_id,
            'data_collected': record,
            'export_date': datetime.now().isoformat(),
            'format': 'JSON',
        }

    def delete_user_data (self, user_id: str) -> Dict:
        """
        Delete all user data (GDPR Article 17 - Right to Erasure).
        """
        if user_id not in self.data_registry:
            return {'error': 'User not found'}

        # Remove from registry
        deleted_record = self.data_registry.pop (user_id)

        # In production: Delete from all systems, databases, backups, logs

        return {
            'deleted': True,
            'user_id': user_id,
            'deletion_date': datetime.now().isoformat(),
            'records_deleted': 1,
        }

    def cleanup_expired_data (self) -> Dict:
        """
        Clean up data past retention period (GDPR Article 5 - Storage Limitation).
        """
        now = datetime.now()
        expired_users = []

        for user_id, record in list (self.data_registry.items()):
            retention_until = datetime.fromisoformat (record['retention_until'])
            if now > retention_until:
                expired_users.append (user_id)
                self.delete_user_data (user_id)

        return {
            'cleanup_date': now.isoformat(),
            'expired_records': len (expired_users),
            'deleted_users': expired_users,
        }

    def _hash_data (self, data: str) -> str:
        """Hash data for storage (don't store raw PII)"""
        return hashlib.sha256(data.encode()).hexdigest()

    def generate_privacy_report (self) -> Dict:
        """Generate privacy compliance report"""
        total_users = len (self.data_registry)
        users_with_pii = sum(
            1 for record in self.data_registry.values()
            if record['pii_detected']
        )

        pii_type_counts = {}
        for record in self.data_registry.values():
            for pii_type in record['pii_types']:
                pii_type_counts[pii_type] = pii_type_counts.get (pii_type, 0) + 1

        return {
            'report_date': datetime.now().isoformat(),
            'total_users': total_users,
            'users_with_pii': users_with_pii,
            'pii_detection_rate': users_with_pii / max (total_users, 1),
            'pii_types_detected': pii_type_counts,
            'avg_retention_days': 30,  # Calculate from records
        }

# Example usage
gdpr_handler = GDPRCompliantPIIHandler()

# Process user data
result = gdpr_handler.process_user_data(
    user_id="user_123",
    data="My email is john@example.com",
    purpose="Customer support inquiry",
    retention_days=30
)
print(f"Processing result: {result}")

# Export user data (right to access)
export = gdpr_handler.export_user_data("user_123")
print(f"\\nUser data export: {export}")

# Delete user data (right to erasure)
deletion = gdpr_handler.delete_user_data("user_123")
print(f"\\nDeletion result: {deletion}")

# Generate privacy report
report = gdpr_handler.generate_privacy_report()
print(f"\\nPrivacy report: {report}")
\`\`\`

## Key Takeaways

1. **Use multiple detection methods**: Regex + NER + custom rules
2. **Implement appropriate redaction**: Full redaction, pseudonymization, or selective
3. **GDPR compliance is mandatory**: Right to access, erasure, portability
4. **Data minimization**: Only collect and retain what's necessary
5. **Audit everything**: Log all PII processing
6. **Regular cleanup**: Delete data past retention periods
7. **Fail safely**: When in doubt, redact

## Production Checklist

- [ ] Multi-method PII detection (regex + NER)
- [ ] Configurable redaction strategies
- [ ] GDPR compliance features (access, erasure, portability)
- [ ] Data retention policies
- [ ] Automated data cleanup
- [ ] PII detection in all system components (input, output, logs)
- [ ] User consent management
- [ ] Privacy reports and auditing
- [ ] Staff training on PII handling
- [ ] Incident response plan for PII leaks
- [ ] Regular privacy impact assessments
- [ ] Third-party vendor compliance

PII protection is not optional—it's a legal requirement and ethical imperative in modern AI systems.
`;
