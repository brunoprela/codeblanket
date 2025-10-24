export const hallucinationDetectionSection = `
# Hallucination Detection

## Introduction

LLM hallucinations—confidently stated false information—are one of the most challenging issues in production AI systems. A hallucinated fact, source, or recommendation can damage user trust, lead to poor decisions, and create liability.

This section covers understanding hallucinations, detecting them through confidence scoring and fact-checking, and implementing mitigation strategies for production systems.

## Understanding Hallucinations

### What are Hallucinations?

LLM hallucinations occur when the model generates information that is:

1. **Factually incorrect**: Wrong dates, numbers, events
2. **Fabricated**: Made-up sources, citations, or quotes
3. **Inconsistent**: Contradicts previous statements
4. **Outdated**: Accurate in training data but no longer true
5. **Contextually wrong**: Technically true but inappropriate for context

### Types of Hallucinations

\`\`\`python
# Examples of different hallucination types

# 1. Factual hallucination
"""
Q: When did World War II end?
A: World War II ended in 1947.  # Wrong, it was 1945
"""

# 2. Source hallucination
"""
Q: What does research say about coffee?
A: According to a 2023 study by Smith et al. in Nature...
# No such study exists
"""

# 3. Calculation hallucination
"""
Q: What is 123 * 456?
A: 123 * 456 = 56,188  # Actually 56,088
"""

# 4. Reasoning hallucination
"""
Q: If all cats are animals and all animals need food, do cats need food?
A: No, cats don't need food because they're independent.
# Logical error
"""

# 5. Temporal hallucination
"""
Q: What's the current president?
A: The current US president is Barack Obama.
# Outdated information
"""
\`\`\`

### Why Hallucinations Happen

1. **Training data limitations**: Model hasn't seen specific information
2. **Overfitting to patterns**: Generates plausible-sounding but false content
3. **Lack of grounding**: No connection to factual knowledge bases
4. **Overconfidence**: Model doesn't know when it doesn't know
5. **Prompt ambiguity**: Unclear questions lead to guessed answers

## Detection Strategies

### Confidence Scoring

\`\`\`python
import openai
from typing import Dict, List, Optional
import json

class ConfidenceScorer:
    """Score LLM output confidence to detect potential hallucinations"""
    
    def __init__(self):
        self.low_confidence_threshold = 0.6
        self.high_confidence_threshold = 0.8
    
    def score_response(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.0
    ) -> Dict:
        """
        Score confidence in an LLM response.
        
        Approach:
        1. Ask the model to rate its own confidence
        2. Generate multiple responses and check consistency
        3. Analyze linguistic hedging
        """
        
        # Method 1: Self-assessment
        self_confidence = self._self_assessment(prompt, response)
        
        # Method 2: Consistency check
        consistency_score = self._consistency_check(prompt, temperature)
        
        # Method 3: Linguistic analysis
        linguistic_score = self._linguistic_analysis(response)
        
        # Combine scores
        overall_confidence = (
            self_confidence * 0.4 +
            consistency_score * 0.4 +
            linguistic_score * 0.2
        )
        
        is_likely_hallucination = overall_confidence < self.low_confidence_threshold
        
        return {
            'overall_confidence': overall_confidence,
            'self_confidence': self_confidence,
            'consistency_score': consistency_score,
            'linguistic_score': linguistic_score,
            'likely_hallucination': is_likely_hallucination,
            'confidence_level': self._confidence_level(overall_confidence)
        }
    
    def _self_assessment(self, prompt: str, response: str) -> float:
        """Ask model to assess its own confidence"""
        
        assessment_prompt = f"""
You previously generated this response to a question.
Now, rate your confidence in the accuracy of this response on a scale of 0.0 to 1.0.

Question: {prompt}
Response: {response}

Respond with ONLY a number between 0.0 and 1.0, where:
- 1.0 = Completely certain, factually verified
- 0.8 = Very confident, likely accurate
- 0.6 = Somewhat confident, may contain errors
- 0.4 = Low confidence, likely contains errors
- 0.2 = Very uncertain, probably hallucinated
- 0.0 = Complete guess, no factual basis

Confidence score:"""
        
        try:
            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": assessment_prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            score_text = result.choices[0].message.content.strip()
            confidence = float(score_text)
            return max(0.0, min(1.0, confidence))
        
        except Exception as e:
            print(f"Self-assessment error: {e}")
            return 0.5  # Default to uncertain
    
    def _consistency_check(
        self,
        prompt: str,
        temperature: float,
        num_samples: int = 5
    ) -> float:
        """
        Generate multiple responses and check consistency.
        High consistency = higher confidence.
        """
        
        responses = []
        for _ in range(num_samples):
            try:
                result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=150
                )
                responses.append(result.choices[0].message.content.strip())
            except Exception as e:
                continue
        
        if len(responses) < 2:
            return 0.5
        
        # Calculate pairwise similarity
        from difflib import SequenceMatcher
        
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = SequenceMatcher(
                    None,
                    responses[i],
                    responses[j]
                ).ratio()
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.5
        return avg_similarity
    
    def _linguistic_analysis(self, response: str) -> float:
        """
        Analyze linguistic patterns that indicate uncertainty.
        
        Hedging words indicate lower confidence:
        - "might", "maybe", "possibly"
        - "I think", "I believe"
        - "could be", "may be"
        """
        
        hedging_words = [
            'might', 'maybe', 'possibly', 'perhaps', 'probably',
            'i think', 'i believe', 'seems like', 'appears to',
            'could be', 'may be', 'not sure', 'uncertain'
        ]
        
        response_lower = response.lower()
        hedging_count = sum(
            1 for word in hedging_words
            if word in response_lower
        )
        
        # More hedging = lower confidence
        # Normalize by response length
        words = response.split()
        hedging_ratio = hedging_count / max(len(words) / 10, 1)
        
        # Convert to confidence score (inverse of hedging)
        confidence = max(0.0, 1.0 - (hedging_ratio * 0.3))
        
        return confidence
    
    def _confidence_level(self, score: float) -> str:
        """Convert numerical score to level"""
        if score >= self.high_confidence_threshold:
            return "high"
        elif score >= self.low_confidence_threshold:
            return "medium"
        else:
            return "low"

# Example usage
scorer = ConfidenceScorer()
prompt = "What is the capital of France?"
response = "I think the capital of France is possibly Paris."
result = scorer.score_response(prompt, response)

print(f"Confidence: {result['overall_confidence']:.2f}")
print(f"Level: {result['confidence_level']}")
print(f"Likely hallucination: {result['likely_hallucination']}")
\`\`\`

### Fact-Checking Integration

\`\`\`python
import requests
from typing import Dict, List, Optional

class FactChecker:
    """Fact-check LLM responses against external sources"""
    
    def __init__(self, serpapi_key: Optional[str] = None):
        self.serpapi_key = serpapi_key
    
    def check_factual_claim(
        self,
        claim: str,
        use_search: bool = True
    ) -> Dict:
        """
        Check a factual claim.
        
        Approaches:
        1. Search for supporting/contradicting sources
        2. Ask another LLM to verify
        3. Check against knowledge base
        """
        
        results = {
            'claim': claim,
            'methods_used': []
        }
        
        # Method 1: Web search verification
        if use_search and self.serpapi_key:
            search_result = self._search_verification(claim)
            results['search_verification'] = search_result
            results['methods_used'].append('search')
        
        # Method 2: LLM cross-verification
        llm_result = self._llm_verification(claim)
        results['llm_verification'] = llm_result
        results['methods_used'].append('llm')
        
        # Combine results
        is_factual = self._combine_verification_results(results)
        results['is_factual'] = is_factual
        results['confidence'] = self._calculate_confidence(results)
        
        return results
    
    def _search_verification(self, claim: str) -> Dict:
        """Verify claim using web search"""
        
        try:
            # Use SerpAPI or similar service
            params = {
                'q': claim,
                'api_key': self.serpapi_key,
                'num': 5
            }
            
            response = requests.get(
                'https://serpapi.com/search',
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                return {'error': 'Search failed', 'verified': False}
            
            data = response.json()
            organic_results = data.get('organic_results', [])
            
            # Simple verification: check if claim appears in results
            supporting_sources = 0
            for result in organic_results[:5]:
                snippet = result.get('snippet', ').lower()
                if any(word in snippet for word in claim.lower().split()):
                    supporting_sources += 1
            
            return {
                'verified': supporting_sources >= 2,
                'supporting_sources': supporting_sources,
                'total_sources': len(organic_results)
            }
        
        except Exception as e:
            return {'error': str(e), 'verified': False}
    
    def _llm_verification(self, claim: str) -> Dict:
        """Use LLM to verify claim"""
        
        verification_prompt = f"""
You are a fact-checker. Analyze the following claim and determine if it's factually accurate.

Claim: {claim}

Respond in JSON format:
{{
    "is_factual": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "requires_verification": true/false
}}
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant."},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            return {
                'error': str(e),
                'is_factual': False,
                'confidence': 0.0
            }
    
    def _combine_verification_results(self, results: Dict) -> bool:
        """Combine multiple verification methods"""
        
        # Search verification
        search_verified = results.get('search_verification', {}).get('verified', False)
        
        # LLM verification
        llm_verified = results.get('llm_verification', {}).get('is_factual', False)
        
        # Both should agree for high confidence
        return search_verified and llm_verified
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence in verification"""
        
        confidences = []
        
        if 'search_verification' in results:
            search_result = results['search_verification']
            if not search_result.get('error'):
                supporting = search_result.get('supporting_sources', 0)
                total = search_result.get('total_sources', 1)
                confidences.append(supporting / max(total, 1))
        
        if 'llm_verification' in results:
            llm_result = results['llm_verification']
            if not llm_result.get('error'):
                confidences.append(llm_result.get('confidence', 0.5))
        
        return sum(confidences) / len(confidences) if confidences else 0.5

# Example usage
fact_checker = FactChecker()
claim = "Paris is the capital of France"
result = fact_checker.check_factual_claim(claim, use_search=False)

print(f"Claim: {claim}")
print(f"Factual: {result['is_factual']}")
print(f"Confidence: {result['confidence']:.2f}")
\`\`\`

### Consistency Validation

\`\`\`python
class ConsistencyValidator:
    """Validate internal consistency of LLM responses"""
    
    def __init__(self):
        pass
    
    def validate_consistency(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> Dict:
        """
        Check if LLM responses are internally consistent.
        
        Args:
            conversation_history: List of {"role": "...", "content": "..."}
        """
        
        assistant_responses = [
            msg['content'] for msg in conversation_history
            if msg['role'] == 'assistant'
        ]
        
        if len(assistant_responses) < 2:
            return {'consistent': True, 'reason': 'Not enough responses to compare'}
        
        # Check for contradictions
        contradictions = self._find_contradictions(assistant_responses)
        
        # Check for fact consistency
        fact_consistency = self._check_fact_consistency(assistant_responses)
        
        is_consistent = len(contradictions) == 0 and fact_consistency
        
        return {
            'consistent': is_consistent,
            'contradictions': contradictions,
            'fact_consistency': fact_consistency,
            'total_responses': len(assistant_responses)
        }
    
    def _find_contradictions(self, responses: List[str]) -> List[Dict]:
        """Find direct contradictions between responses"""
        
        # Use LLM to find contradictions
        contradiction_prompt = f"""
Analyze these responses for contradictions:

{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(responses))}

Are there any contradictory statements? Respond in JSON:
{{
    "has_contradictions": true/false,
    "contradictions": [
        {{"response_1": 1, "response_2": 2, "contradiction": "description"}}
    ]
}}
"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": contradiction_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('contradictions', [])
        
        except Exception as e:
            return []
    
    def _check_fact_consistency(self, responses: List[str]) -> bool:
        """Check if facts mentioned are consistent"""
        
        # Extract entities and facts from each response
        # Check if same entities have consistent attributes
        
        # Simplified version: check if numbers are consistent
        import re
        
        numbers_by_response = []
        for response in responses:
            numbers = re.findall(r'\\b\\d+\\b', response)
            numbers_by_response.append(set(numbers))
        
        # If same context, numbers should be similar
        # This is a simplified check
        if len(numbers_by_response) < 2:
            return True
        
        # Check for significant differences
        all_numbers = set().union(*numbers_by_response)
        if len(all_numbers) > 10:  # Too many different numbers might indicate inconsistency
            return False
        
        return True

# Example usage
validator = ConsistencyValidator()
conversation = [
    {"role": "user", "content": "What's the population of Tokyo?"},
    {"role": "assistant", "content": "Tokyo has about 14 million people."},
    {"role": "user", "content": "And how many people live in Tokyo?"},
    {"role": "assistant", "content": "Tokyo's population is around 9 million."},  # Inconsistent!
]

result = validator.validate_consistency(conversation)
print(f"Consistent: {result['consistent']}")
print(f"Contradictions: {result['contradictions']}")
\`\`\`

## Mitigation Strategies

### Uncertainty Expressions

\`\`\`python
class UncertaintyExpr esser:
    """Add uncertainty expressions when confidence is low"""
    
    def __init__(self):
        self.confidence_scorer = ConfidenceScorer()
    
    def add_uncertainty(
        self,
        prompt: str,
        response: str
    ) -> str:
        """Add uncertainty expressions to low-confidence responses"""
        
        # Score confidence
        confidence_result = self.confidence_scorer.score_response(prompt, response)
        confidence = confidence_result['overall_confidence']
        
        if confidence >= 0.8:
            # High confidence: no modification needed
            return response
        
        elif confidence >= 0.6:
            # Medium confidence: mild hedging
            prefixes = [
                "Based on available information, ",
                "It appears that ",
                "Generally, "
            ]
            return prefixes[0] + response
        
        elif confidence >= 0.4:
            # Low confidence: strong hedging
            prefixes = [
                "I'm not entirely certain, but ",
                "This information may not be fully accurate, but ",
                "To the best of my knowledge, "
            ]
            return prefixes[0] + response
        
        else:
            # Very low confidence: disclaimer
            disclaimer = (
                "⚠️ Low confidence warning: I'm very uncertain about this information. "
                "Please verify independently. "
            )
            return disclaimer + response

# Example usage
expresser = UncertaintyExpresser()
prompt = "What's the GDP of Uzbekistan in 2023?"
response = "The GDP of Uzbekistan in 2023 is $80 billion."

# This is likely a hallucination or outdated, so add uncertainty
modified = expresser.add_uncertainty(prompt, response)
print(modified)
\`\`\`

### Grounded Generation

\`\`\`python
class GroundedGenerator:
    """Generate responses grounded in provided context"""
    
    def __init__(self):
        pass
    
    def generate_grounded(
        self,
        query: str,
        context: str,
        require_citations: bool = True
    ) -> Dict:
        """
        Generate response grounded in provided context.
        
        Reduces hallucinations by:
        1. Providing explicit context
        2. Instructing to use only provided information
        3. Requiring citations
        """
        
        system_prompt = """
You are an assistant that answers questions based ONLY on the provided context.

RULES:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Never make up or infer information not in the context
4. {citation_rule}
"""
        
        citation_rule = (
            "Cite the specific part of the context you used"
            if require_citations
            else "Provide direct quotes when possible"
        )
        
        system_prompt = system_prompt.format(citation_rule=citation_rule)
        
        user_prompt = f"""
Context:
{context}

Question: {query}

Answer (based only on the context above):"""
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            answer = response.choices[0].message.content
            
            # Verify answer is grounded in context
            is_grounded = self._verify_grounding(answer, context)
            
            return {
                'answer': answer,
                'is_grounded': is_grounded,
                'context_used': context,
                'method': 'grounded_generation'
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'answer': None,
                'is_grounded': False
            }
    
    def _verify_grounding(self, answer: str, context: str) -> bool:
        """Verify that answer is grounded in context"""
        
        # Simple check: key phrases from answer should appear in context
        # More sophisticated: use NLI model
        
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Check overlap
        overlap = len(answer_words.intersection(context_words))
        overlap_ratio = overlap / max(len(answer_words), 1)
        
        # Should have significant overlap
        return overlap_ratio > 0.5

# Example usage
generator = GroundedGenerator()
context = """
Paris is the capital and most populous city of France.
The city has a population of approximately 2.1 million.
"""

query = "What is the capital of France?"
result = generator.generate_grounded(query, context)

print(f"Answer: {result['answer']}")
print(f"Grounded: {result['is_grounded']}")
\`\`\`

## Comprehensive Hallucination Detection System

\`\`\`python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class HallucinationDetectionResult:
    """Result of hallucination detection"""
    likely_hallucination: bool
    confidence: float
    reasons: List[str]
    detection_methods: List[str]
    suggested_action: str
    modified_response: Optional[str]

class HallucinationDetectionSystem:
    """
    Comprehensive hallucination detection combining:
    1. Confidence scoring
    2. Fact-checking
    3. Consistency validation
    4. Grounding verification
    """
    
    def __init__(self):
        self.confidence_scorer = ConfidenceScorer()
        self.fact_checker = FactChecker()
        self.consistency_validator = ConsistencyValidator()
        self.uncertainty_expresser = UncertaintyExpresser()
    
    def detect(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None,
        check_facts: bool = False
    ) -> HallucinationDetectionResult:
        """
        Comprehensive hallucination detection.
        
        Args:
            prompt: User prompt
            response: LLM response
            context: Optional context for grounding check
            conversation_history: Optional conversation for consistency
            check_facts: Whether to perform fact-checking (slow)
        """
        
        reasons = []
        methods = []
        confidence_scores = []
        
        # Method 1: Confidence scoring
        methods.append('confidence_scoring')
        confidence_result = self.confidence_scorer.score_response(prompt, response)
        confidence_scores.append(confidence_result['overall_confidence'])
        
        if confidence_result['likely_hallucination']:
            reasons.append(f"Low confidence score: {confidence_result['overall_confidence']:.2f}")
        
        # Method 2: Consistency validation
        if conversation_history:
            methods.append('consistency_validation')
            consistency_result = self.consistency_validator.validate_consistency(
                conversation_history + [{"role": "assistant", "content": response}]
            )
            
            if not consistency_result['consistent']:
                reasons.append("Inconsistent with previous responses")
                confidence_scores.append(0.3)
        
        # Method 3: Fact-checking (optional, slow)
        if check_facts:
            methods.append('fact_checking')
            # Extract claims and check them
            # Simplified: check entire response
            fact_result = self.fact_checker.check_factual_claim(response, use_search=False)
            
            if not fact_result.get('is_factual', True):
                reasons.append("Fact-checking failed")
                confidence_scores.append(fact_result.get('confidence', 0.5))
        
        # Make final decision
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        likely_hallucination = avg_confidence < 0.6 or len(reasons) > 0
        
        # Suggest action
        if likely_hallucination:
            if avg_confidence < 0.4:
                suggested_action = "block"
            elif avg_confidence < 0.6:
                suggested_action = "add_uncertainty"
            else:
                suggested_action = "flag_for_review"
        else:
            suggested_action = "allow"
        
        # Modify response if needed
        modified_response = None
        if suggested_action == "add_uncertainty":
            modified_response = self.uncertainty_expresser.add_uncertainty(prompt, response)
        
        return HallucinationDetectionResult(
            likely_hallucination=likely_hallucination,
            confidence=avg_confidence,
            reasons=reasons if reasons else ["No hallucination indicators detected"],
            detection_methods=methods,
            suggested_action=suggested_action,
            modified_response=modified_response
        )

# Example usage
detector = HallucinationDetectionSystem()

# Test case 1: Likely factual
prompt1 = "What is 2 + 2?"
response1 = "2 + 2 = 4"
result1 = detector.detect(prompt1, response1)
print(f"\\nTest 1:")
print(f"Likely hallucination: {result1.likely_hallucination}")
print(f"Confidence: {result1.confidence:.2f}")
print(f"Action: {result1.suggested_action}")

# Test case 2: Likely hallucination
prompt2 = "When was Einstein born?"
response2 = "Einstein was born in 1885."  # Actually 1879
result2 = detector.detect(prompt2, response2)
print(f"\\nTest 2:")
print(f"Likely hallucination: {result2.likely_hallucination}")
print(f"Confidence: {result2.confidence:.2f}")
print(f"Reasons: {result2.reasons}")
print(f"Action: {result2.suggested_action}")
\`\`\`

## Key Takeaways

1. **Confidence scoring**: Use multiple methods to assess confidence
2. **Fact-checking**: Verify claims against external sources when critical
3. **Consistency validation**: Check for internal contradictions
4. **Grounded generation**: Provide context to reduce hallucinations
5. **Uncertainty expressions**: Add hedging when confidence is low
6. **User warnings**: Alert users when information may be inaccurate
7. **Continuous monitoring**: Track hallucination rates

## Production Checklist

- [ ] Confidence scoring implemented
- [ ] Fact-checking for critical domains
- [ ] Consistency validation across conversations
- [ ] Grounded generation with citations
- [ ] Uncertainty expressions for low-confidence outputs
- [ ] User warnings and disclaimers
- [ ] Hallucination rate monitoring
- [ ] Feedback mechanism for users to report inaccuracies
- [ ] Regular evaluation of detection accuracy
- [ ] Documentation of known limitations
- [ ] A/B testing of mitigation strategies

Hallucination detection is imperfect—combine multiple methods and always provide users with ways to verify critical information.
`;
