export const auditLoggingComplianceSection = `
# Audit Logging & Compliance

## Introduction

Audit logging is essential for compliance (GDPR, CCPA, HIPAA, SOC 2), incident investigation, and demonstrating accountability. Every AI interaction should be logged, traceable, and auditable.

This section covers building comprehensive audit systems, meeting regulatory requirements, and implementing compliance frameworks for production AI systems.

## Why Audit Logging Matters

### Regulatory Requirements

Different regulations require different logging:

\`\`\`python
# GDPR Requirements (EU)
gdpr_requirements = {
    'data_processing_records': 'Log all personal data processing',
    'consent_tracking': 'Track user consent for data processing',
    'right_to_access': 'Provide users with all their data',
    'right_to_erasure': 'Delete user data on request',
    'data_breach_notification': 'Log and report breaches within 72 hours',
    'retention_limits': 'Delete data after retention period'
}

# CCPA Requirements (California)
ccpa_requirements = {
    'data_collection_disclosure': 'Disclose what data is collected',
    'right_to_know': 'Tell users what data you have about them',
    'right_to_delete': 'Delete personal information on request',
    'right_to_opt_out': 'Allow opt-out of data selling',
    'non_discrimination': "Don\'t discriminate against users who exercise rights"
}

# HIPAA Requirements (Healthcare, US)
hipaa_requirements = {
    'phi_protection': 'Protect Protected Health Information',
    'access_logging': 'Log all PHI access',
    'audit_controls': 'Implement audit controls',
    'data_encryption': 'Encrypt PHI in transit and at rest',
    'breach_notification': 'Notify affected individuals of breaches'
}

# SOC 2 Requirements
soc2_requirements = {
    'security': 'System is protected against unauthorized access',
    'availability': 'System is available for operation',
    'processing_integrity': 'System processing is complete, valid, accurate',
    'confidentiality': 'Confidential information is protected',
    'privacy': 'Personal information is collected, used, and disclosed properly'
}
\`\`\`

### Use Cases for Audit Logs

1. **Incident Investigation**: What happened? Who was affected?
2. **Compliance Audits**: Demonstrate regulatory compliance
3. **Security Analysis**: Detect breaches and attacks
4. **User Support**: Help users understand what happened
5. **System Debugging**: Diagnose technical issues
6. **Usage Analytics**: Understand system usage patterns
7. **Cost Attribution**: Track costs per user/team

## Comprehensive Audit Logging System

\`\`\`python
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib

class EventType(Enum):
    """Types of events to log"""
    USER_REQUEST = "user_request"
    LLM_RESPONSE = "llm_response"
    SAFETY_VIOLATION = "safety_violation"
    PII_DETECTED = "pii_detected"
    RATE_LIMIT = "rate_limit"
    AUTH_ATTEMPT = "auth_attempt"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    CONSENT_GIVEN = "consent_given"
    CONSENT_REVOKED = "consent_revoked"
    SYSTEM_ERROR = "system_error"

class Severity(Enum):
    """Event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents an audit event"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: Severity
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    action: str
    resource: Optional[str]
    details: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, storage_backend=None):
        """
        Args:
            storage_backend: Database or file system for storing logs
                            (Use PostgreSQL, MongoDB, or ELK in production)
        """
        self.storage = storage_backend or InMemoryStorage()
        self.pii_redactor = PIIRedactor()  # From earlier section
    
    def log_event(
        self,
        event_type: EventType,
        action: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: Severity = Severity.INFO
    ) -> AuditEvent:
        """Log an audit event"""
        
        # Generate unique event ID
        event_id = self._generate_event_id()
        
        # Redact PII from details
        safe_details = self._redact_pii_from_dict(details or {})
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            details=safe_details,
            success=success,
            error_message=error_message
        )
        
        # Store event
        self.storage.store(event)
        
        # Alert on critical events
        if severity == Severity.CRITICAL:
            self._send_alert(event)
        
        return event
    
    def log_user_request(
        self,
        user_id: str,
        request_content: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log a user request"""
        
        # Redact PII from content before logging
        safe_content, _ = self.pii_redactor.redact(request_content)
        
        self.log_event(
            event_type=EventType.USER_REQUEST,
            action="submit_prompt",
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource="llm",
            details={
                'content_hash': self._hash_content(request_content),
                'content_length': len(request_content),
                'metadata': metadata or {}
            },
            success=True
        )
    
    def log_llm_response(
        self,
        user_id: str,
        request_hash: str,
        response_content: str,
        model: str,
        tokens_used: int,
        cost: float,
        latency_ms: float,
        session_id: Optional[str] = None
    ):
        """Log an LLM response"""
        
        # Redact PII
        safe_response, _ = self.pii_redactor.redact(response_content)
        
        self.log_event(
            event_type=EventType.LLM_RESPONSE,
            action="generate_response",
            user_id=user_id,
            session_id=session_id,
            resource="llm",
            details={
                'request_hash': request_hash,
                'response_hash': self._hash_content(response_content),
                'response_length': len(response_content),
                'model': model,
                'tokens_used': tokens_used,
                'cost': cost,
                'latency_ms': latency_ms
            },
            success=True
        )
    
    def log_safety_violation(
        self,
        user_id: str,
        violation_type: str,
        content_hash: str,
        severity: Severity,
        details: Dict
    ):
        """Log a safety violation"""
        
        self.log_event(
            event_type=EventType.SAFETY_VIOLATION,
            action=f"blocked_{violation_type}",
            user_id=user_id,
            resource="safety_system",
            details={
                'violation_type': violation_type,
                'content_hash': content_hash,
                **details
            },
            success=True,
            severity=severity
        )
    
    def log_data_access(
        self,
        user_id: str,
        accessed_user_id: str,
        data_type: str,
        reason: str,
        ip_address: Optional[str] = None
    ):
        """Log data access (for GDPR/HIPAA compliance)"""
        
        self.log_event(
            event_type=EventType.DATA_ACCESS,
            action=f"access_{data_type}",
            user_id=user_id,
            ip_address=ip_address,
            resource=f"user_data:{accessed_user_id}",
            details={
                'accessed_user': accessed_user_id,
                'data_type': data_type,
                'reason': reason
            },
            success=True,
            severity=Severity.WARNING  # Data access should be monitored
        )
    
    def log_consent(
        self,
        user_id: str,
        consent_type: str,
        granted: bool,
        details: Optional[Dict] = None
    ):
        """Log user consent (GDPR requirement)"""
        
        self.log_event(
            event_type=EventType.CONSENT_GIVEN if granted else EventType.CONSENT_REVOKED,
            action=f"consent_{consent_type}",
            user_id=user_id,
            details={
                'consent_type': consent_type,
                'granted': granted,
                **(details or {})
            },
            success=True
        )
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _hash_content(self, content: str) -> str:
        """Hash content for reference without storing sensitive data"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _redact_pii_from_dict(self, data: Dict) -> Dict:
        """Redact PII from dictionary"""
        # Simple implementation - in production, recursively redact
        safe_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                safe_value, _ = self.pii_redactor.redact(value)
                safe_data[key] = safe_value
            else:
                safe_data[key] = value
        return safe_data
    
    def _send_alert(self, event: AuditEvent):
        """Send alert for critical events"""
        # In production: send to PagerDuty, Slack, email, etc.
        print(f"🚨 CRITICAL ALERT: {event.action}")
        print(f"   Details: {event.details}")

class InMemoryStorage:
    """In-memory storage for development (use database in production)"""
    
    def __init__(self):
        self.events: List[AuditEvent] = []
    
    def store(self, event: AuditEvent):
        """Store event"""
        self.events.append(event)
    
    def query(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Query events"""
        results = self.events
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        
        return results

# Example usage
logger = AuditLogger()

# Log user request
logger.log_user_request(
    user_id="user_123",
    request_content="What's the weather in Paris?",
    session_id="session_456",
    ip_address="192.168.1.1"
)

# Log LLM response
logger.log_llm_response(
    user_id="user_123",
    request_hash="abc123",
    response_content="The weather in Paris is sunny, 22°C.",
    model="gpt-4",
    tokens_used=150,
    cost=0.003,
    latency_ms=250.0,
    session_id="session_456"
)

# Log safety violation
logger.log_safety_violation(
    user_id="user_789",
    violation_type="prompt_injection",
    content_hash="xyz789",
    severity=Severity.CRITICAL,
    details={'pattern': 'instruction_override'}
)

print(f"\\nLogged {len(logger.storage.events)} events")
\`\`\`

## GDPR Compliance Implementation

\`\`\`python
class GDPRCompliance System:
    """GDPR compliance features"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.data_registry: Dict[str, Dict] = {}  # User data inventory
    
    def register_data_processing(
        self,
        user_id: str,
        data_type: str,
        purpose: str,
        legal_basis: str,
        retention_days: int
    ):
        """
        Register data processing activity (GDPR Article 30).
        
        Args:
            user_id: User whose data is processed
            data_type: Type of data (e.g., 'prompt', 'response', 'pii')
            purpose: Purpose of processing
            legal_basis: Legal basis (consent, contract, legitimate interest, etc.)
            retention_days: How long data will be retained
        """
        
        if user_id not in self.data_registry:
            self.data_registry[user_id] = {
                'user_id': user_id,
                'processing_activities': [],
                'consents': {},
                'created_at': datetime.now().isoformat()
            }
        
        activity = {
            'data_type': data_type,
            'purpose': purpose,
            'legal_basis': legal_basis,
            'retention_days': retention_days,
            'retention_until': (datetime.now() + timedelta(days=retention_days)).isoformat(),
            'registered_at': datetime.now().isoformat()
        }
        
        self.data_registry[user_id]['processing_activities'].append(activity)
        
        # Log registration
        self.audit_logger.log_event(
            event_type=EventType.DATA_ACCESS,
            action="register_data_processing",
            user_id=user_id,
            details=activity,
            success=True
        )
    
    def export_user_data(self, user_id: str) -> Dict:
        """
        Export all user data (GDPR Article 20 - Right to Data Portability).
        
        Returns structured, machine-readable format of all user data.
        """
        
        # Log access
        self.audit_logger.log_data_access(
            user_id="system",
            accessed_user_id=user_id,
            data_type="all",
            reason="user_data_export_request"
        )
        
        # Collect all data
        user_data = {
            'user_id': user_id,
            'export_date': datetime.now().isoformat(),
            'data_processing_activities': self.data_registry.get(user_id, {}),
            'audit_log': self._get_user_audit_log(user_id),
            'format': 'JSON',
            'regulation': 'GDPR Article 20'
        }
        
        return user_data
    
    def delete_user_data(self, user_id: str, reason: str) -> Dict:
        """
        Delete all user data (GDPR Article 17 - Right to Erasure).
        
        Args:
            user_id: User whose data to delete
            reason: Reason for deletion
        
        Returns:
            Deletion confirmation
        """
        
        # Log deletion request
        self.audit_logger.log_event(
            event_type=EventType.DATA_DELETION,
            action="delete_all_user_data",
            user_id=user_id,
            details={'reason': reason},
            success=True,
            severity=Severity.WARNING
        )
        
        # Delete from registry
        deleted_data = self.data_registry.pop(user_id, None)
        
        # In production: Delete from all systems
        # - Database records
        # - Log files
        # - Backups (mark for deletion)
        # - Third-party systems
        
        return {
            'deleted': True,
            'user_id': user_id,
            'deletion_date': datetime.now().isoformat(),
            'data_deleted': 'all',
            'reason': reason,
            'records_deleted': len(deleted_data['processing_activities']) if deleted_data else 0
        }
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for purpose"""
        
        user_data = self.data_registry.get(user_id, {})
        consents = user_data.get('consents', {})
        
        return consents.get(purpose, False)
    
    def record_consent(
        self,
        user_id: str,
        purpose: str,
        granted: bool,
        consent_method: str
    ):
        """Record user consent"""
        
        if user_id not in self.data_registry:
            self.data_registry[user_id] = {
                'user_id': user_id,
                'processing_activities': [],
                'consents': {},
                'created_at': datetime.now().isoformat()
            }
        
        self.data_registry[user_id]['consents'][purpose] = {
            'granted': granted,
            'method': consent_method,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log consent
        self.audit_logger.log_consent(
            user_id=user_id,
            consent_type=purpose,
            granted=granted,
            details={'method': consent_method}
        )
    
    def _get_user_audit_log(self, user_id: str) -> List[Dict]:
        """Get all audit log events for user"""
        
        events = self.audit_logger.storage.query(user_id=user_id)
        return [event.to_dict() for event in events]
    
    def generate_compliance_report(self) -> Dict:
        """Generate GDPR compliance report"""
        
        total_users = len(self.data_registry)
        users_with_consent = sum(
            1 for user in self.data_registry.values()
            if any(c['granted'] for c in user.get('consents', {}).values())
        )
        
        return {
            'report_date': datetime.now().isoformat(),
            'total_users': total_users,
            'users_with_consent': users_with_consent,
            'consent_rate': users_with_consent / max(total_users, 1),
            'data_retention_policy': 'In place',
            'right_to_access': 'Implemented',
            'right_to_erasure': 'Implemented',
            'data_portability': 'Implemented',
            'audit_logging': 'Active'
        }

# Example usage
compliance = GDPRComplianceSystem(audit_logger=logger)

# Register data processing
compliance.register_data_processing(
    user_id="user_123",
    data_type="llm_prompts",
    purpose="service_provision",
    legal_basis="consent",
    retention_days=30
)

# Record consent
compliance.record_consent(
    user_id="user_123",
    purpose="service_provision",
    granted=True,
    consent_method="opt_in_checkbox"
)

# Export user data
export = compliance.export_user_data("user_123")
print(f"\\nExported data for user: {export['user_id']}")
print(f"Processing activities: {len(export['data_processing_activities'].get('processing_activities', []))}")

# Generate compliance report
report = compliance.generate_compliance_report()
print(f"\\nCompliance report:")
print(f"  Total users: {report['total_users']}")
print(f"  Consent rate: {report['consent_rate']:.0%}")
\`\`\`

## Security & Access Controls

\`\`\`python
from functools import wraps

class AuditAccessControl:
    """Access control for audit logs"""
    
    def __init__(self):
        self.role_permissions = {
            'admin': {'read', 'write', 'delete', 'export'},
            'security': {'read', 'export'},
            'support': {'read'},
            'developer': {'read'},
            'user': set()  # Users can only access their own data
        }
    
    def require_permission(self, permission: str):
        """Decorator to require permission for function"""
        def decorator(func):
            @wraps(func)
            def wrapper(self, user_role: str, *args, **kwargs):
                if permission not in self.role_permissions.get(user_role, set()):
                    raise PermissionError(
                        f"Role '{user_role}' does not have '{permission}' permission"
                    )
                return func(self, user_role, *args, **kwargs)
            return wrapper
        return decorator
    
    @require_permission('export')
    def export_audit_logs(
        self,
        user_role: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Export audit logs (requires export permission)"""
        # Implementation here
        pass
    
    @require_permission('delete')
    def delete_audit_logs(
        self,
        user_role: str,
        event_ids: List[str]
    ):
        """Delete audit logs (requires delete permission)"""
        # This should rarely be used and heavily logged
        pass
\`\`\`

## Key Takeaways

1. **Log everything**: All interactions, access, modifications
2. **GDPR/CCPA compliance**: Right to access, erasure, portability
3. **Secure storage**: Encrypt logs, restrict access
4. **Retention policies**: Delete logs after retention period
5. **PII redaction**: Don't store PII in logs unnecessarily
6. **Audit trail**: Immutable, tamper-proof logs
7. **Regular audits**: Review logs for compliance

## Production Checklist

- [ ] Comprehensive audit logging implemented
- [ ] GDPR compliance features (access, erasure, portability)
- [ ] Consent management system
- [ ] Data retention policies
- [ ] PII redaction in logs
- [ ] Secure log storage (encrypted)
- [ ] Access controls for logs
- [ ] Regular compliance audits
- [ ] Incident response procedures
- [ ] User data export functionality
- [ ] Automated retention cleanup
- [ ] Compliance documentation
- [ ] Regular penetration testing
- [ ] Third-party compliance audits

Audit logging is your proof of compliance—invest in a robust system from day one.
`;
