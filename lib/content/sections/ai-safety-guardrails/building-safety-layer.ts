export const buildingSafetyLayerSection = `
# Building a Safety Layer

## Introduction

A production safety layer integrates all safety components—content moderation, prompt injection defense, PII detection, output validation, rate limiting, bias detection, and audit logging—into a cohesive, defense-in-depth system.

This section covers architecting a complete safety layer, implementing pre/post-processing checks, human review workflows, and building safety-first AI applications.

## Safety Layer Architecture

### Layered Defense Strategy

\`\`\`python
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class SafetyCheckResult(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    BLOCK = "block"

@dataclass
class SafetyLayerResult:
    """Result from safety layer processing"""
    allowed: bool
    reason: str
    modified_input: Optional[str]
    modified_output: Optional[str]
    safety_scores: Dict[str, float]
    checks_performed: List[str]
    violations: List[Dict]
    should_flag: bool
    should_review: bool

class ComprehensiveSafetyLayer:
    """
    Comprehensive safety layer with multiple defense layers:

    Layer 1: Input Pre-Processing
    - Rate limiting
    - Authentication/authorization
    - Input sanitization
    - PII detection
    - Prompt injection detection

    Layer 2: Request Processing
    - Content moderation
    - Bias detection
    - Context validation

    Layer 3: Generation Controls
    - Safe prompt construction
    - Model selection
    - Temperature/parameter controls

    Layer 4: Output Post-Processing
    - Output validation
    - Hallucination detection
    - PII redaction
    - Bias correction
    - Content moderation

    Layer 5: Audit & Monitoring
    - Comprehensive logging
    - Metrics collection
    - Alerting
    """

    def __init__(self):
        # Layer 1: Input checks
        self.rate_limiter = ProductionSafetyRateLimiter()
        self.injection_detector = InjectionDefenseSystem()
        self.input_pii_detector = ComprehensivePIIDetector()

        # Layer 2: Content checks
        self.content_moderator = MultiLevelModerator()
        self.bias_detector = TextBiasDetector()

        # Layer 3: Generation
        self.prompt_engineer = FairPromptEngineer()

        # Layer 4: Output checks
        self.output_validator = ComprehensiveValidator()
        self.hallucination_detector = HallucinationDetectionSystem()
        self.output_pii_detector = ComprehensivePIIDetector()
        self.bias_corrector = BiasCorrector()

        # Layer 5: Audit
        self.audit_logger = AuditLogger()
        self.metrics_collector = MetricsCollector()

    def process_request(
        self,
        user_id: str,
        prompt: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> SafetyLayerResult:
        """
        Process request through all safety layers.

        Returns result indicating if request should be allowed,
        any modifications made, and safety metadata.
        """

        safety_scores = {}
        checks_performed = []
        violations = []

        try:
            # ============================================
            # LAYER 1: INPUT PRE-PROCESSING
            # ============================================

            # Check 1: Rate limiting
            checks_performed.append('rate_limiting')
            rate_limit_result = self.rate_limiter.check_request(
                user_id=user_id,
                request_content=prompt
            )

            if not rate_limit_result.allowed:
                self.audit_logger.log_safety_violation(
                    user_id=user_id,
                    violation_type='rate_limit',
                    content_hash=self._hash (prompt),
                    severity=Severity.WARNING,
                    details={'reason': rate_limit_result.reason}
                )

                return SafetyLayerResult(
                    allowed=False,
                    reason=f"Rate limit exceeded: {rate_limit_result.reason}",
                    modified_input=None,
                    modified_output=None,
                    safety_scores={'rate_limit': 0.0},
                    checks_performed=checks_performed,
                    violations=[{'type': 'rate_limit', 'reason': rate_limit_result.reason}],
                    should_flag=False,
                    should_review=False
                )

            safety_scores['rate_limit'] = 1.0

            # Check 2: Prompt injection
            checks_performed.append('injection_detection')
            injection_result = self.injection_detector.defend (prompt)

            if injection_result.should_block:
                violations.append({
                    'type': 'prompt_injection',
                    'reason': injection_result.reason,
                    'risk_level': injection_result.risk_level
                })

                self.audit_logger.log_safety_violation(
                    user_id=user_id,
                    violation_type='prompt_injection',
                    content_hash=self._hash (prompt),
                    severity=Severity.CRITICAL,
                    details={'detections': injection_result.detections}
                )

                return SafetyLayerResult(
                    allowed=False,
                    reason="Prompt injection detected",
                    modified_input=None,
                    modified_output=None,
                    safety_scores={'injection': 0.0},
                    checks_performed=checks_performed,
                    violations=violations,
                    should_flag=True,
                    should_review=False
                )

            # Use sanitized input
            safe_prompt = injection_result.sanitized_input
            safety_scores['injection'] = 1.0

            # Check 3: Input PII detection
            checks_performed.append('input_pii')
            input_pii_result = self.input_pii_detector.detect (safe_prompt)

            if input_pii_result.has_pii:
                violations.append({
                    'type': 'pii_in_input',
                    'pii_types': list (input_pii_result.pii_types),
                    'count': len (input_pii_result.pii_found)
                })

                # Redact PII
                redactor = PIIRedactor()
                safe_prompt, _ = redactor.redact (safe_prompt)

                self.audit_logger.log_safety_violation(
                    user_id=user_id,
                    violation_type='pii_detected',
                    content_hash=self._hash (prompt),
                    severity=Severity.WARNING,
                    details={'pii_types': list (input_pii_result.pii_types)}
                )

            safety_scores['input_pii'] = 0.5 if input_pii_result.has_pii else 1.0

            # ============================================
            # LAYER 2: CONTENT MODERATION
            # ============================================

            # Check 4: Content moderation
            checks_performed.append('content_moderation')
            moderation_result = self.content_moderator.moderate (safe_prompt)

            if moderation_result.should_block:
                violations.append({
                    'type': 'content_moderation',
                    'level': moderation_result.level.name,
                    'reasons': moderation_result.reasons
                })

                self.audit_logger.log_safety_violation(
                    user_id=user_id,
                    violation_type='content_violation',
                    content_hash=self._hash (prompt),
                    severity=Severity.HIGH,
                    details=moderation_result.details
                )

                return SafetyLayerResult(
                    allowed=False,
                    reason=f"Content moderation failed: {', '.join (moderation_result.reasons)}",
                    modified_input=None,
                    modified_output=None,
                    safety_scores={'moderation': 0.0},
                    checks_performed=checks_performed,
                    violations=violations,
                    should_flag=True,
                    should_review=False
                )

            safety_scores['moderation'] = 1.0

            # Check 5: Input bias detection
            checks_performed.append('input_bias')
            bias_detections = self.bias_detector.detect_bias (safe_prompt)
            if bias_detections:
                violations.append({
                    'type': 'input_bias',
                    'detections': [d.bias_type for d in bias_detections]
                })

            safety_scores['input_bias'] = 0.7 if bias_detections else 1.0

            # ============================================
            # LAYER 3: SAFE GENERATION
            # ============================================

            # Construct safe prompt with fairness instructions
            checks_performed.append('safe_prompt_construction')
            safe_prompt_with_instructions = self.prompt_engineer.make_fair_prompt (safe_prompt)

            # Log successful pre-processing
            self.audit_logger.log_user_request(
                user_id=user_id,
                request_content=safe_prompt,
                session_id=session_id,
                ip_address=ip_address,
                metadata={
                    'checks_performed': checks_performed,
                    'safety_scores': safety_scores,
                    'violations': violations
                }
            )

            # At this point, input has passed all checks
            # Return with safe prompt for generation
            return SafetyLayerResult(
                allowed=True,
                reason="All input safety checks passed",
                modified_input=safe_prompt_with_instructions,
                modified_output=None,
                safety_scores=safety_scores,
                checks_performed=checks_performed,
                violations=violations,
                should_flag=len (violations) > 0,
                should_review=False
            )

        except Exception as e:
            # Log error
            self.audit_logger.log_event(
                event_type=EventType.SYSTEM_ERROR,
                action="safety_layer_error",
                user_id=user_id,
                error_message=str (e),
                severity=Severity.ERROR,
                success=False
            )

            # Fail safely: block on error
            return SafetyLayerResult(
                allowed=False,
                reason=f"Safety layer error: {str (e)}",
                modified_input=None,
                modified_output=None,
                safety_scores={},
                checks_performed=checks_performed,
                violations=[{'type': 'system_error', 'error': str (e)}],
                should_flag=True,
                should_review=True
            )

    def process_response(
        self,
        user_id: str,
        original_prompt: str,
        response: str,
        request_hash: str,
        model: str,
        tokens_used: int,
        cost: float,
        latency_ms: float
    ) -> SafetyLayerResult:
        """Process LLM response through output safety checks"""

        safety_scores = {}
        checks_performed = []
        violations = []
        modified_response = response

        try:
            # ============================================
            # LAYER 4: OUTPUT POST-PROCESSING
            # ============================================

            # Check 1: Output PII detection
            checks_performed.append('output_pii')
            output_pii_result = self.output_pii_detector.detect (response)

            if output_pii_result.has_pii:
                violations.append({
                    'type': 'pii_in_output',
                    'pii_types': list (output_pii_result.pii_types),
                    'count': len (output_pii_result.pii_found)
                })

                # Redact PII from output
                redactor = PIIRedactor()
                modified_response, _ = redactor.redact (modified_response)

                self.audit_logger.log_safety_violation(
                    user_id=user_id,
                    violation_type='pii_in_output',
                    content_hash=self._hash (response),
                    severity=Severity.CRITICAL,
                    details={'pii_types': list (output_pii_result.pii_types)}
                )

            safety_scores['output_pii'] = 0.0 if output_pii_result.has_pii else 1.0

            # Check 2: Output content moderation
            checks_performed.append('output_moderation')
            output_moderation = self.content_moderator.moderate (modified_response)

            if output_moderation.should_block:
                violations.append({
                    'type': 'output_content_violation',
                    'reasons': output_moderation.reasons
                })

                self.audit_logger.log_safety_violation(
                    user_id=user_id,
                    violation_type='output_content_violation',
                    content_hash=self._hash (response),
                    severity=Severity.HIGH,
                    details=output_moderation.details
                )

                # Block output
                return SafetyLayerResult(
                    allowed=False,
                    reason="Output failed content moderation",
                    modified_input=None,
                    modified_output=None,
                    safety_scores=safety_scores,
                    checks_performed=checks_performed,
                    violations=violations,
                    should_flag=True,
                    should_review=True
                )

            safety_scores['output_moderation'] = 1.0

            # Check 3: Hallucination detection
            checks_performed.append('hallucination')
            hallucination_result = self.hallucination_detector.detect(
                prompt=original_prompt,
                response=modified_response,
                check_facts=False  # Too slow for real-time
            )

            if hallucination_result.likely_hallucination:
                violations.append({
                    'type': 'hallucination',
                    'confidence': hallucination_result.confidence,
                    'reasons': hallucination_result.reasons
                })

                # Add uncertainty if needed
                if hallucination_result.modified_response:
                    modified_response = hallucination_result.modified_response

            safety_scores['hallucination'] = hallucination_result.confidence

            # Check 4: Output bias detection and correction
            checks_performed.append('output_bias')
            bias_detections = self.bias_detector.detect_bias (modified_response)

            if bias_detections:
                violations.append({
                    'type': 'output_bias',
                    'detections': [d.bias_type for d in bias_detections]
                })

                # Correct bias
                modified_response, corrections = self.bias_corrector.correct_bias (modified_response)

            safety_scores['output_bias'] = 0.7 if bias_detections else 1.0

            # Check 5: Output validation
            checks_performed.append('output_validation')
            validation_result = self.output_validator.validate(
                output=modified_response,
                prompt=original_prompt
            )

            if not validation_result.is_valid:
                violations.append({
                    'type': 'output_validation',
                    'issues': [i.message for i in validation_result.issues]
                })

            safety_scores['output_validation'] = validation_result.quality_score

            # ============================================
            # LAYER 5: AUDIT & METRICS
            # ============================================

            # Log response
            self.audit_logger.log_llm_response(
                user_id=user_id,
                request_hash=request_hash,
                response_content=modified_response,
                model=model,
                tokens_used=tokens_used,
                cost=cost,
                latency_ms=latency_ms
            )

            # Collect metrics
            self.metrics_collector.record_response(
                user_id=user_id,
                safety_scores=safety_scores,
                violations=violations,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost=cost
            )

            # Determine if response should be reviewed
            critical_violations = [v for v in violations if v['type'] in ['pii_in_output', 'hallucination']]
            should_review = len (critical_violations) > 0 or sum (safety_scores.values()) / len (safety_scores) < 0.7

            # Calculate overall safety score
            overall_score = sum (safety_scores.values()) / len (safety_scores) if safety_scores else 0.0

            return SafetyLayerResult(
                allowed=True,
                reason="Output safety checks completed",
                modified_input=None,
                modified_output=modified_response if modified_response != response else None,
                safety_scores=safety_scores,
                checks_performed=checks_performed,
                violations=violations,
                should_flag=len (violations) > 0,
                should_review=should_review
            )

        except Exception as e:
            # Log error
            self.audit_logger.log_event(
                event_type=EventType.SYSTEM_ERROR,
                action="output_safety_error",
                user_id=user_id,
                error_message=str (e),
                severity=Severity.ERROR,
                success=False
            )

            # Fail safely: block on error
            return SafetyLayerResult(
                allowed=False,
                reason=f"Output safety error: {str (e)}",
                modified_input=None,
                modified_output=None,
                safety_scores={},
                checks_performed=checks_performed,
                violations=[{'type': 'system_error', 'error': str (e)}],
                should_flag=True,
                should_review=True
            )

    def _hash (self, content: str) -> str:
        """Hash content for logging"""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class MetricsCollector:
    """Collect safety metrics"""

    def __init__(self):
        self.metrics = defaultdict (list)

    def record_response(
        self,
        user_id: str,
        safety_scores: Dict[str, float],
        violations: List[Dict],
        latency_ms: float,
        tokens_used: int,
        cost: float
    ):
        """Record response metrics"""
        self.metrics['safety_scores'].append (safety_scores)
        self.metrics['violations'].append (len (violations))
        self.metrics['latency'].append (latency_ms)
        self.metrics['tokens'].append (tokens_used)
        self.metrics['cost'].append (cost)

    def get_dashboard (self) -> Dict:
        """Get safety dashboard metrics"""
        return {
            'total_requests': len (self.metrics['safety_scores']),
            'avg_safety_score': self._avg_overall_score(),
            'violation_rate': sum(1 for v in self.metrics['violations'] if v > 0) / max (len (self.metrics['violations']), 1),
            'avg_latency_ms': sum (self.metrics['latency']) / max (len (self.metrics['latency']), 1),
            'total_cost': sum (self.metrics['cost'])
        }

    def _avg_overall_score (self) -> float:
        """Calculate average overall safety score"""
        if not self.metrics['safety_scores']:
            return 0.0

        total_score = 0.0
        count = 0

        for scores in self.metrics['safety_scores']:
            if scores:
                total_score += sum (scores.values()) / len (scores)
                count += 1

        return total_score / max (count, 1)

# Example usage
safety_layer = ComprehensiveSafetyLayer()

# Process request
input_result = safety_layer.process_request(
    user_id="user_123",
    prompt="Tell me about Paris",
    session_id="session_456"
)

print(f"Input allowed: {input_result.allowed}")
print(f"Checks performed: {', '.join (input_result.checks_performed)}")
print(f"Safety scores: {input_result.safety_scores}")
print(f"Violations: {len (input_result.violations)}")

if input_result.allowed:
    # Simulate LLM call
    llm_response = "Paris is the capital of France, known for the Eiffel Tower."

    # Process response
    output_result = safety_layer.process_response(
        user_id="user_123",
        original_prompt="Tell me about Paris",
        response=llm_response,
        request_hash="abc123",
        model="gpt-4",
        tokens_used=150,
        cost=0.003,
        latency_ms=250.0
    )

    print(f"\\nOutput allowed: {output_result.allowed}")
    print(f"Output safety scores: {output_result.safety_scores}")
    print(f"Should review: {output_result.should_review}")
\`\`\`

## Human-in-the-Loop Review

\`\`\`python
from typing import List, Dict
from enum import Enum

class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"

@dataclass
class ReviewItem:
    """Item queued for human review"""
    item_id: str
    timestamp: datetime
    user_id: str
    content_type: str  # 'input' or 'output'
    content: str
    safety_result: SafetyLayerResult
    priority: str  # 'low', 'medium', 'high', 'critical'
    status: ReviewStatus
    reviewer_id: Optional[str]
    reviewer_notes: Optional[str]
    decision_timestamp: Optional[datetime]

class HumanReviewQueue:
    """Queue for human review of flagged content"""

    def __init__(self):
        self.queue: List[ReviewItem] = []

    def add_for_review(
        self,
        user_id: str,
        content: str,
        safety_result: SafetyLayerResult,
        content_type: str = 'output'
    ) -> ReviewItem:
        """Add item to review queue"""

        # Determine priority
        priority = self._calculate_priority (safety_result)

        item = ReviewItem(
            item_id=str (uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            content_type=content_type,
            content=content,
            safety_result=safety_result,
            priority=priority,
            status=ReviewStatus.PENDING,
            reviewer_id=None,
            reviewer_notes=None,
            decision_timestamp=None
        )

        self.queue.append (item)

        # Sort by priority
        self.queue.sort (key=lambda x: {
            'critical': 0, 'high': 1, 'medium': 2, 'low': 3
        }[x.priority])

        return item

    def get_next_for_review (self) -> Optional[ReviewItem]:
        """Get next item for review"""
        pending = [item for item in self.queue if item.status == ReviewStatus.PENDING]
        return pending[0] if pending else None

    def submit_review(
        self,
        item_id: str,
        reviewer_id: str,
        decision: ReviewStatus,
        notes: str
    ):
        """Submit review decision"""

        for item in self.queue:
            if item.item_id == item_id:
                item.status = decision
                item.reviewer_id = reviewer_id
                item.reviewer_notes = notes
                item.decision_timestamp = datetime.now()
                break

    def _calculate_priority (self, safety_result: SafetyLayerResult) -> str:
        """Calculate review priority"""

        # Check for critical violations
        critical_types = {'pii_in_output', 'prompt_injection', 'content_violation'}
        has_critical = any(
            v['type'] in critical_types
            for v in safety_result.violations
        )

        if has_critical:
            return 'critical'

        # Check safety scores
        if safety_result.safety_scores:
            avg_score = sum (safety_result.safety_scores.values()) / len (safety_result.safety_scores)
            if avg_score < 0.5:
                return 'high'
            elif avg_score < 0.7:
                return 'medium'

        return 'low'

    def get_queue_status (self) -> Dict:
        """Get queue status"""
        return {
            'total_items': len (self.queue),
            'pending': sum(1 for item in self.queue if item.status == ReviewStatus.PENDING),
            'approved': sum(1 for item in self.queue if item.status == ReviewStatus.APPROVED),
            'rejected': sum(1 for item in self.queue if item.status == ReviewStatus.REJECTED),
            'by_priority': {
                'critical': sum(1 for item in self.queue if item.priority == 'critical' and item.status == ReviewStatus.PENDING),
                'high': sum(1 for item in self.queue if item.priority == 'high' and item.status == ReviewStatus.PENDING),
                'medium': sum(1 for item in self.queue if item.priority == 'medium' and item.status == ReviewStatus.PENDING),
                'low': sum(1 for item in self.queue if item.priority == 'low' and item.status == ReviewStatus.PENDING),
            }
        }
\`\`\`

## Key Takeaways

1. **Defense in depth**: Multiple layers of protection
2. **Pre and post-processing**: Check inputs AND outputs
3. **Comprehensive logging**: Audit everything
4. **Human oversight**: Review flagged content
5. **Fail safely**: Block on errors or uncertainty
6. **Metrics and monitoring**: Track safety metrics
7. **Continuous improvement**: Learn from violations

## Production Checklist

- [ ] All safety components integrated
- [ ] Input pre-processing (rate limit, injection, PII, moderation)
- [ ] Output post-processing (PII, moderation, hallucination, bias)
- [ ] Comprehensive audit logging
- [ ] Human review queue
- [ ] Safety metrics dashboard
- [ ] Alerting for critical violations
- [ ] Incident response procedures
- [ ] Regular safety audits
- [ ] A/B testing of safety measures
- [ ] Documentation for team
- [ ] User communication about safety

A complete safety layer is not optional—it's the foundation of responsible AI in production.
`;
