export const contentModerationSection = `
# Content Moderation

## Introduction

Content moderation is the first line of defense against harmful outputs in AI systems. Whether you're building a chatbot, code generator, or content creation tool, you need robust systems to detect and block toxic, harmful, NSFW, and dangerous content before it reaches users.

This section covers implementing production-grade content moderation using OpenAI's moderation API, building custom filters, handling edge cases, and creating multi-level moderation systems that balance safety with usability.

## Why Content Moderation Matters

### Real-World Consequences

Unmoderated AI systems have led to:

- **Brand damage**: AI chatbots generating offensive or inappropriate content
- **Legal liability**: Platforms hosting AI-generated harmful content
- **User harm**: Exposure to traumatic, disturbing, or triggering content
- **Regulatory violations**: Failure to comply with content safety laws
- **Platform bans**: Violation of Terms of Service for cloud providers

### Production Requirements

A production content moderation system must:

1. **Detect multiple content types**: Hate speech, violence, sexual content, self-harm
2. **Operate in real-time**: Low latency for user-facing applications
3. **Handle multiple languages**: Global applications need multilingual support
4. **Balance safety and usability**: Minimize false positives while catching violations
5. **Provide explanations**: Users and moderators need to understand decisions
6. **Scale efficiently**: Handle thousands of requests per second
7. **Comply with regulations**: Meet legal requirements for different jurisdictions

## OpenAI Moderation API

### Basic Implementation

\`\`\`python
import openai
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ModerationResult:
    """Structured moderation result"""
    flagged: bool
    categories: Dict[str, bool]
    category_scores: Dict[str, float]
    explanation: str

def moderate_content (text: str) -> ModerationResult:
    """
    Use OpenAI's moderation API to check content safety.
    
    Categories checked:
    - hate: Hate speech
    - hate/threatening: Hateful with violence
    - harassment: Harassment
    - harassment/threatening: Harassment with violence
    - self-harm: Self-harm content
    - self-harm/intent: Intent to self-harm
    - self-harm/instructions: Instructions for self-harm
    - sexual: Sexual content
    - sexual/minors: Sexual content involving minors
    - violence: Violence
    - violence/graphic: Graphic violence
    """
    
    try:
        response = openai.Moderation.create (input=text)
        result = response["results"][0]
        
        # Determine explanation
        if result["flagged"]:
            flagged_categories = [
                cat for cat, flagged in result["categories"].items()
                if flagged
            ]
            explanation = f"Content flagged for: {', '.join (flagged_categories)}"
        else:
            explanation = "Content passed all moderation checks"
        
        return ModerationResult(
            flagged=result["flagged"],
            categories=result["categories"],
            category_scores=result["category_scores"],
            explanation=explanation
        )
    
    except Exception as e:
        # On error, fail safely by flagging content
        return ModerationResult(
            flagged=True,
            categories={},
            category_scores={},
            explanation=f"Moderation error: {str (e)}"
        )

# Example usage
text = "I want to hurt someone"
result = moderate_content (text)

if result.flagged:
    print(f"❌ Content blocked: {result.explanation}")
    print(f"Scores: {result.category_scores}")
else:
    print("✅ Content approved")
\`\`\`

### Custom Thresholds

The default thresholds may not fit your use case. Implement custom thresholds:

\`\`\`python
from typing import Dict, Tuple

class CustomModerationEngine:
    """Content moderation with custom thresholds"""
    
    def __init__(self):
        # Custom thresholds per category (0.0 to 1.0)
        self.thresholds = {
            'hate': 0.3,                    # More strict
            'hate/threatening': 0.2,        # Very strict
            'harassment': 0.4,
            'harassment/threatening': 0.2,  # Very strict
            'self-harm': 0.1,               # Extremely strict
            'self-harm/intent': 0.1,
            'self-harm/instructions': 0.05,  # Most strict
            'sexual': 0.5,                  # Context-dependent
            'sexual/minors': 0.0,           # Zero tolerance
            'violence': 0.3,
            'violence/graphic': 0.2,
        }
    
    def moderate (self, text: str) -> Tuple[bool, Dict]:
        """
        Moderate content with custom thresholds.
        
        Returns:
            (is_safe, details)
        """
        response = openai.Moderation.create (input=text)
        result = response["results"][0]
        
        violations = []
        scores_exceeded = {}
        
        for category, score in result["category_scores"].items():
            threshold = self.thresholds.get (category, 0.5)
            
            if score > threshold:
                violations.append (category)
                scores_exceeded[category] = {
                    'score': score,
                    'threshold': threshold,
                    'exceeded_by': score - threshold
                }
        
        is_safe = len (violations) == 0
        
        details = {
            'safe': is_safe,
            'violations': violations,
            'scores': result["category_scores"],
            'scores_exceeded': scores_exceeded,
            'flagged_by_api': result["flagged"],
            'action': 'allow' if is_safe else 'block'
        }
        
        return is_safe, details

# Example usage
moderator = CustomModerationEngine()
text = "This is a borderline inappropriate comment"
is_safe, details = moderator.moderate (text)

if not is_safe:
    print(f"❌ Blocked: {details['violations']}")
    for cat, info in details['scores_exceeded'].items():
        print(f"  {cat}: {info['score']:.3f} (threshold: {info['threshold']:.3f})")
\`\`\`

### Context-Aware Moderation

Different contexts require different moderation levels:

\`\`\`python
from enum import Enum

class ContentContext(Enum):
    """Different content contexts requiring different moderation"""
    CHILD_SAFE = "child_safe"          # Strictest (e.g., kids' apps)
    GENERAL_AUDIENCE = "general"       # Strict (e.g., public platforms)
    TEEN_PLUS = "teen_plus"            # Moderate (e.g., teen social media)
    MATURE = "mature"                  # Relaxed (e.g., adult content)
    MEDICAL = "medical"                # Special handling (e.g., health apps)
    EDUCATIONAL = "educational"        # Special handling (e.g., learning)

class ContextAwareModerator:
    """Moderation with context-specific thresholds"""
    
    def __init__(self):
        # Define threshold sets for each context
        self.context_thresholds = {
            ContentContext.CHILD_SAFE: {
                'hate': 0.1,
                'harassment': 0.1,
                'self-harm': 0.05,
                'sexual': 0.0,  # Zero tolerance
                'violence': 0.1,
            },
            ContentContext.GENERAL_AUDIENCE: {
                'hate': 0.3,
                'harassment': 0.3,
                'self-harm': 0.1,
                'sexual': 0.2,
                'violence': 0.3,
            },
            ContentContext.MATURE: {
                'hate': 0.5,
                'harassment': 0.5,
                'self-harm': 0.2,
                'sexual': 0.7,  # More permissive
                'violence': 0.5,
            },
            ContentContext.EDUCATIONAL: {
                # Allow discussion of difficult topics in educational context
                'hate': 0.4,
                'harassment': 0.5,
                'self-harm': 0.6,  # Can discuss for awareness
                'sexual': 0.5,     # Sex education
                'violence': 0.5,   # Historical context
            }
        }
    
    def moderate(
        self,
        text: str,
        context: ContentContext
    ) -> Dict:
        """Moderate with context-aware thresholds"""
        
        response = openai.Moderation.create (input=text)
        result = response["results"][0]
        
        thresholds = self.context_thresholds[context]
        violations = []
        
        for category, score in result["category_scores"].items():
            # Get base category (remove sub-categories)
            base_category = category.split('/')[0]
            threshold = thresholds.get (base_category, 0.5)
            
            if score > threshold:
                violations.append({
                    'category': category,
                    'score': score,
                    'threshold': threshold,
                    'context': context.value
                })
        
        return {
            'safe': len (violations) == 0,
            'context': context.value,
            'violations': violations,
            'all_scores': result["category_scores"]
        }

# Example usage
moderator = ContextAwareModerator()

# Same text, different contexts
text = "The violent revolution led to many deaths"

# Educational context: Likely allowed (discussing history)
edu_result = moderator.moderate (text, ContentContext.EDUCATIONAL)
print(f"Educational: {'✅ Allowed' if edu_result['safe'] else '❌ Blocked'}")

# Child-safe context: Likely blocked
child_result = moderator.moderate (text, ContentContext.CHILD_SAFE)
print(f"Child-safe: {'✅ Allowed' if child_result['safe'] else '❌ Blocked'}")
\`\`\`

## Custom Content Filters

### Keyword Blocklists

\`\`\`python
import re
from typing import List, Set, Dict

class KeywordFilter:
    """Keyword-based content filtering"""
    
    def __init__(self):
        # Exact match blocklist
        self.blocked_words: Set[str] = set()
        
        # Pattern-based blocklist (regex)
        self.blocked_patterns: List[re.Pattern] = []
        
        # Contextual blocklist (requires surrounding words)
        self.contextual_blocks: Dict[str, List[str]] = {}
        
        self.load_blocklists()
    
    def load_blocklists (self):
        """Load blocklists from configuration"""
        
        # Exact matches (case-insensitive)
        self.blocked_words = {
            # Add your blocklist here
            # Note: Real blocklists are much more comprehensive
        }
        
        # Pattern-based blocks
        self.blocked_patterns = [
            re.compile (r'\\b(?:spam|scam)\\b', re.IGNORECASE),
            re.compile (r'\\b(?:hack|crack)\\s+(?:password|account)\\b', re.IGNORECASE),
            # Add more patterns
        ]
        
        # Contextual blocks (word + context)
        self.contextual_blocks = {
            'kill': ['yourself', 'himself', 'herself'],  # Block "kill yourself" but not "kill the process"
            # Add more contextual rules
        }
    
    def check (self, text: str) -> Dict:
        """Check text against keyword filters"""
        
        text_lower = text.lower()
        violations = []
        
        # Check exact matches
        words = set (re.findall (r'\\b\\w+\\b', text_lower))
        blocked_found = words.intersection (self.blocked_words)
        if blocked_found:
            violations.extend([
                {'type': 'blocked_word', 'word': word}
                for word in blocked_found
            ])
        
        # Check patterns
        for pattern in self.blocked_patterns:
            matches = pattern.findall (text)
            if matches:
                violations.extend([
                    {'type': 'blocked_pattern', 'match': match}
                    for match in matches
                ])
        
        # Check contextual blocks
        for trigger, contexts in self.contextual_blocks.items():
            if trigger in text_lower:
                for context in contexts:
                    if context in text_lower:
                        violations.append({
                            'type': 'contextual_block',
                            'trigger': trigger,
                            'context': context
                        })
        
        return {
            'safe': len (violations) == 0,
            'violations': violations,
            'filter_type': 'keyword'
        }
\`\`\`

### Machine Learning-Based Filters

\`\`\`python
from transformers import pipeline
from typing import Dict, List

class MLContentFilter:
    """ML-based content filtering using Hugging Face models"""
    
    def __init__(self):
        # Toxicity detection model
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1  # CPU, use 0 for GPU
        )
        
        # Hate speech detection
        self.hate_classifier = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=-1
        )
        
        self.thresholds = {
            'toxicity': 0.7,
            'hate_speech': 0.6,
        }
    
    def check_toxicity (self, text: str) -> Dict:
        """Check for toxic content"""
        
        try:
            result = self.toxicity_classifier (text)[0]
            score = result['score'] if result['label'] == 'toxic' else 1 - result['score']
            
            is_toxic = score > self.thresholds['toxicity']
            
            return {
                'is_toxic': is_toxic,
                'score': score,
                'label': 'toxic' if is_toxic else 'non-toxic',
                'confidence': result['score']
            }
        except Exception as e:
            # Fail safely
            return {
                'is_toxic': True,
                'score': 1.0,
                'label': 'error',
                'error': str (e)
            }
    
    def check_hate_speech (self, text: str) -> Dict:
        """Check for hate speech"""
        
        try:
            result = self.hate_classifier (text)[0]
            
            is_hate = (
                result['label'] == 'hate'
                and result['score'] > self.thresholds['hate_speech']
            )
            
            return {
                'is_hate_speech': is_hate,
                'label': result['label'],
                'score': result['score'],
            }
        except Exception as e:
            # Fail safely
            return {
                'is_hate_speech': True,
                'label': 'error',
                'score': 1.0,
                'error': str (e)
            }
    
    def moderate (self, text: str) -> Dict:
        """Comprehensive ML-based moderation"""
        
        toxicity = self.check_toxicity (text)
        hate = self.check_hate_speech (text)
        
        violations = []
        if toxicity['is_toxic']:
            violations.append({
                'type': 'toxicity',
                'score': toxicity['score']
            })
        
        if hate['is_hate_speech']:
            violations.append({
                'type': 'hate_speech',
                'score': hate['score']
            })
        
        return {
            'safe': len (violations) == 0,
            'violations': violations,
            'toxicity': toxicity,
            'hate_speech': hate
        }

# Example usage
ml_filter = MLContentFilter()
text = "You're an idiot and I hate you"
result = ml_filter.moderate (text)

if not result['safe']:
    print(f"❌ Content blocked:")
    for violation in result['violations']:
        print(f"  - {violation['type']}: {violation['score']:.2f}")
\`\`\`

## Multi-Level Moderation System

\`\`\`python
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class ModerationLevel(Enum):
    """Moderation severity levels"""
    ALLOW = 1
    FLAG = 2
    REVIEW = 3
    BLOCK = 4

@dataclass
class ModerationDecision:
    """Final moderation decision"""
    level: ModerationLevel
    reasons: List[str]
    scores: Dict[str, float]
    details: Dict
    should_block: bool
    should_flag: bool
    should_review: bool

class MultiLevelModerator:
    """
    Multi-level moderation combining multiple approaches:
    1. OpenAI Moderation API
    2. Custom keyword filters
    3. ML-based toxicity detection
    4. Pattern matching
    """
    
    def __init__(self):
        self.openai_moderator = CustomModerationEngine()
        self.keyword_filter = KeywordFilter()
        self.ml_filter = MLContentFilter()
    
    def moderate(
        self,
        text: str,
        context: Optional[ContentContext] = None
    ) -> ModerationDecision:
        """
        Perform multi-level moderation.
        
        Decision logic:
        - BLOCK: Any critical violation
        - REVIEW: Medium confidence violations
        - FLAG: Low confidence violations
        - ALLOW: No violations detected
        """
        
        reasons = []
        all_scores = {}
        details = {}
        
        # Level 1: OpenAI Moderation (fastest, most reliable)
        openai_safe, openai_details = self.openai_moderator.moderate (text)
        details['openai'] = openai_details
        all_scores.update (openai_details['scores'])
        
        if not openai_safe:
            reasons.append (f"OpenAI flagged: {', '.join (openai_details['violations'])}")
            # Critical violations = immediate block
            if any (v.startswith('self-harm') or v == 'sexual/minors' 
                   for v in openai_details['violations']):
                return ModerationDecision(
                    level=ModerationLevel.BLOCK,
                    reasons=reasons,
                    scores=all_scores,
                    details=details,
                    should_block=True,
                    should_flag=False,
                    should_review=False
                )
        
        # Level 2: Keyword filter (fast, rule-based)
        keyword_result = self.keyword_filter.check (text)
        details['keywords'] = keyword_result
        
        if not keyword_result['safe']:
            reasons.append (f"Blocked keywords: {len (keyword_result['violations'])}")
            return ModerationDecision(
                level=ModerationLevel.BLOCK,
                reasons=reasons,
                scores=all_scores,
                details=details,
                should_block=True,
                should_flag=False,
                should_review=False
            )
        
        # Level 3: ML-based detection (slower, more nuanced)
        ml_result = self.ml_filter.moderate (text)
        details['ml'] = ml_result
        
        if not ml_result['safe']:
            for violation in ml_result['violations']:
                if violation['score'] > 0.9:
                    # High confidence = block
                    reasons.append (f"High {violation['type']}: {violation['score']:.2f}")
                    return ModerationDecision(
                        level=ModerationLevel.BLOCK,
                        reasons=reasons,
                        scores=all_scores,
                        details=details,
                        should_block=True,
                        should_flag=False,
                        should_review=False
                    )
                elif violation['score'] > 0.7:
                    # Medium confidence = review
                    reasons.append (f"Possible {violation['type']}: {violation['score']:.2f}")
                    return ModerationDecision(
                        level=ModerationLevel.REVIEW,
                        reasons=reasons,
                        scores=all_scores,
                        details=details,
                        should_block=False,
                        should_flag=False,
                        should_review=True
                    )
                else:
                    # Low confidence = flag
                    reasons.append (f"Low {violation['type']}: {violation['score']:.2f}")
        
        # Level 4: Final decision
        if reasons:
            return ModerationDecision(
                level=ModerationLevel.FLAG,
                reasons=reasons,
                scores=all_scores,
                details=details,
                should_block=False,
                should_flag=True,
                should_review=False
            )
        
        # All checks passed
        return ModerationDecision(
            level=ModerationLevel.ALLOW,
            reasons=["All moderation checks passed"],
            scores=all_scores,
            details=details,
            should_block=False,
            should_flag=False,
            should_review=False
        )
\`\`\`

## Production Moderation Pipeline

\`\`\`python
import asyncio
from typing import Dict, List
import time

class ProductionModerationPipeline:
    """Production-ready moderation with caching, rate limiting, and monitoring"""
    
    def __init__(self):
        self.moderator = MultiLevelModerator()
        self.cache = {}  # Use Redis in production
        self.metrics = ModerationMetrics()
    
    async def moderate_async(
        self,
        text: str,
        user_id: str,
        context: Optional[ContentContext] = None
    ) -> ModerationDecision:
        """Asynchronous moderation with full production features"""
        
        start_time = time.time()
        
        # Check cache
        cache_key = self.get_cache_key (text, context)
        cached_result = self.cache.get (cache_key)
        if cached_result:
            self.metrics.record_cache_hit()
            return cached_result
        
        # Perform moderation
        try:
            decision = self.moderator.moderate (text, context)
            
            # Cache result (cache for 1 hour)
            self.cache[cache_key] = decision
            
            # Record metrics
            self.metrics.record_moderation(
                decision=decision,
                duration=time.time() - start_time,
                user_id=user_id
            )
            
            # Log if flagged or blocked
            if decision.should_block or decision.should_flag:
                await self.log_violation (user_id, text, decision)
            
            return decision
            
        except Exception as e:
            # On error, fail safely by blocking
            self.metrics.record_error (str (e))
            return ModerationDecision(
                level=ModerationLevel.BLOCK,
                reasons=[f"Moderation error: {str (e)}"],
                scores={},
                details={'error': str (e)},
                should_block=True,
                should_flag=False,
                should_review=False
            )
    
    def get_cache_key (self, text: str, context: Optional[ContentContext]) -> str:
        """Generate cache key"""
        import hashlib
        context_str = context.value if context else 'default'
        return hashlib.sha256(f"{text}:{context_str}".encode()).hexdigest()
    
    async def log_violation(
        self,
        user_id: str,
        text: str,
        decision: ModerationDecision
    ):
        """Log moderation violations for review"""
        # In production, log to database or logging service
        print(f"⚠️  Violation logged for user {user_id}:")
        print(f"   Level: {decision.level.name}")
        print(f"   Reasons: {', '.join (decision.reasons)}")

class ModerationMetrics:
    """Track moderation metrics"""
    
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.blocks = 0
        self.flags = 0
        self.reviews = 0
        self.errors = 0
        self.total_duration = 0.0
    
    def record_moderation(
        self,
        decision: ModerationDecision,
        duration: float,
        user_id: str
    ):
        """Record moderation metrics"""
        self.total_requests += 1
        self.total_duration += duration
        
        if decision.should_block:
            self.blocks += 1
        elif decision.should_flag:
            self.flags += 1
        elif decision.should_review:
            self.reviews += 1
    
    def record_cache_hit (self):
        """Record cache hit"""
        self.total_requests += 1
        self.cache_hits += 1
    
    def record_error (self, error: str):
        """Record error"""
        self.errors += 1
        print(f"❌ Moderation error: {error}")
    
    def get_stats (self) -> Dict:
        """Get moderation statistics"""
        return {
            'total_requests': self.total_requests,
            'cache_hit_rate': self.cache_hits / max (self.total_requests, 1),
            'block_rate': self.blocks / max (self.total_requests, 1),
            'flag_rate': self.flags / max (self.total_requests, 1),
            'review_rate': self.reviews / max (self.total_requests, 1),
            'error_rate': self.errors / max (self.total_requests, 1),
            'avg_duration': self.total_duration / max (self.total_requests, 1),
        }
\`\`\`

## Handling False Positives

\`\`\`python
class FalsePositiveHandler:
    """Handle and learn from false positives"""
    
    def __init__(self):
        self.allowlist = set()
        self.false_positive_log = []
    
    def report_false_positive(
        self,
        text: str,
        original_decision: ModerationDecision,
        user_id: str,
        reviewer_notes: str
    ):
        """Report a false positive for learning"""
        
        self.false_positive_log.append({
            'text': text,
            'original_decision': original_decision,
            'user_id': user_id,
            'reviewer_notes': reviewer_notes,
            'timestamp': time.time()
        })
        
        # Add to allowlist if appropriate
        if self._should_allowlist (text, reviewer_notes):
            self.allowlist.add (text.lower())
    
    def _should_allowlist (self, text: str, notes: str) -> bool:
        """Determine if text should be allowlisted"""
        # Add logic to decide if this specific text should be allowed
        # E.g., if reviewer says "medical terminology" or "educational context"
        allowlist_keywords = ['medical', 'educational', 'technical', 'historical']
        return any (kw in notes.lower() for kw in allowlist_keywords)
    
    def check_allowlist (self, text: str) -> bool:
        """Check if text is in allowlist"""
        return text.lower() in self.allowlist
    
    def analyze_false_positives (self) -> Dict:
        """Analyze false positive patterns"""
        if not self.false_positive_log:
            return {'total': 0}
        
        # Group by violation type
        by_type = {}
        for fp in self.false_positive_log:
            for reason in fp['original_decision'].reasons:
                by_type[reason] = by_type.get (reason, 0) + 1
        
        return {
            'total': len (self.false_positive_log),
            'by_type': by_type,
            'allowlist_size': len (self.allowlist)
        }
\`\`\`

## Key Takeaways

1. **Use multiple detection methods**: API + keywords + ML
2. **Context matters**: Adjust thresholds based on use case
3. **Fail safely**: When uncertain, block rather than allow
4. **Cache aggressively**: Moderation is expensive
5. **Learn from mistakes**: Track and handle false positives
6. **Monitor continuously**: Track block rates and patterns
7. **Provide explanations**: Users need to know why content was blocked

## Production Checklist

- [ ] OpenAI Moderation API integrated
- [ ] Custom thresholds configured for your use case
- [ ] Keyword blocklists maintained
- [ ] ML-based filters for nuanced detection
- [ ] Multi-level decision logic implemented
- [ ] Caching for performance
- [ ] Metrics and monitoring
- [ ] False positive handling
- [ ] User appeal process
- [ ] Regular audits of moderation decisions
- [ ] Compliance with platform policies
- [ ] Documentation for moderators

Content moderation is not "one and done"—it requires continuous monitoring, tuning, and improvement as usage patterns and attack vectors evolve.
`;
