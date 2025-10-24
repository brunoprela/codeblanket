/**
 * Quiz questions for Hallucination Detection section
 */

export const hallucinationdetectionQuiz = [
  {
    id: 'halluc-det-q-1',
    question:
      'Your LLM-based research assistant generates responses with 30% containing hallucinated citations (fake studies, wrong authors). Design a hallucination detection and mitigation system. What techniques would you use for detection, and how would you prevent hallucinations from reaching users?',
    hint: 'Consider confidence scoring, fact-checking, and citation verification.',
    sampleAnswer:
      '**Problem:** 30% hallucination rate in citations. Unacceptable for research tool. **Multi-Layer Detection:** **Layer 1: Confidence Scoring** - Ask model to rate confidence: def get_confidence(response): confidence_prompt = f"Rate your confidence in this response (0.0-1.0): {response}". confidence = llm(confidence_prompt). return float(confidence). If confidence < 0.7: Add disclaimer or block. **Layer 2: Citation Extraction & Verification** - Extract citations: citations = extract_citations(response)  # "Smith et al., 2023 in Nature". Verify each: for citation in citations: if not verify_citation_exists(citation): hallucination_detected = True. Verification via: Crossref API, Google Scholar API, PubMed, or arXiv. **Layer 3: Consistency Check** - Generate response multiple times (temperature=0.0), compare: responses = [generate() for _ in range(3)]. if all_similar(responses): consistent = True. else: likely_hallucination = True. **Layer 4: Fact-Checking** - Use retrieval to ground response: docs = search_academic_databases(query). response = generate_grounded(query, docs). Verify response only uses info from docs. **Prevention:** System prompt: "CRITICAL: Only cite papers you are certain exist. If uncertain, say \'I don\'t have verified information.\'" Output validation: Block responses with unverified citations. User warnings: "Citations should be independently verified." **Result:** 30% → 5% hallucination rate.',
    keyPoints: [
      'Confidence scoring identifies uncertain responses',
      'Citation verification catches fake references',
      'Consistency checking detects hallucinations',
      'Grounded generation with retrieval prevents hallucinations',
    ],
  },
  {
    id: 'halluc-det-q-2',
    question:
      'Design a real-time hallucination scoring system that runs in production with <100ms added latency. Your current fact-checking approach (external API calls) takes 2 seconds. How do you optimize for production speed while maintaining detection accuracy?',
    hint: 'Consider fast heuristics, caching, and async processing.',
    sampleAnswer:
      '**Problem:** Fact-checking: 2 seconds. Target: <100ms. **Fast Detection Methods:** **Method 1: Linguistic Cues (5ms)** - Detect hedging words indicating uncertainty: hedging = ["might", "possibly", "I think", "perhaps",]. count = sum(1 for word in hedging if word in response). if count > 3: confidence_score = 0.6. Fast, no API calls. **Method 2: Self-Consistency (Async, 0ms perceived)** - Generate 3 responses in parallel during initial generation. Compare after. No additional user-facing latency. **Method 3: Confidence from Logprobs (0ms)** - Use token probabilities from LLM: if avg_token_probability < 0.7: likely_uncertain. OpenAI/Anthropic provide logprobs. **Method 4: Semantic Caching (2ms)** - Cache fact-check results: if similar_query_in_cache(query): return cached_hallucination_score. 80% cache hit rate → Most requests 2ms. **Method 5: Fast Pattern Matching (3ms)** - Known hallucination patterns: if "according to studies" but no specific citation: flag. if specific numbers without source: flag. **Combined Real-Time System:** def detect_hallucination_fast(response): score = 1.0. # Linguistic cues (5ms), hedging_count = count_hedging(response). score -= hedging_count * 0.1. # Pattern matching (3ms), if has_unsourced_claims(response): score -= 0.2. # Confidence from generation (0ms), if avg_logprob < threshold: score -= 0.3. return score  # Total: ~10ms. **Async Fact-Checking:** Fast score for immediate response. Detailed fact-check in background. If hallucination found within 30s: Flag retroactively, notify user. **Result:** User sees response in <100ms. Full fact-check completes in 2s (background).',
    keyPoints: [
      'Use fast heuristics: linguistic cues, patterns',
      'Cache fact-check results',
      'Async background verification',
      'Combine multiple fast methods for real-time scoring',
    ],
  },
  {
    id: 'halluc-det-q-3',
    question:
      'Your hallucination detector has 85% recall (catches 85% of hallucinations) and 90% precision. A medical chatbot cannot tolerate false negatives (missed hallucinations). How do you optimize for higher recall while managing the increase in false positives?',
    hint: 'Consider conservative thresholds and human review for flagged content.',
    sampleAnswer:
      '**Problem:** Medical chatbot. Hallucinations = patient harm. Current: 85% recall (misses 15% of hallucinations). Need: >98% recall. **Trade-off:** Higher recall → Lower threshold → More false positives. **Optimization:** **Strategy 1: Lower Confidence Threshold** - Current: Block if hallucination_score > 0.7. New: Block if hallucination_score > 0.5. Result: Recall 85% → 95% (catch more), Precision 90% → 70% (more false positives). **Strategy 2: Multiple Detection Methods** - Combine: Confidence scoring, Citation verification, Consistency checking, Medical knowledge base lookup. Block if ANY method flags: Recall increases (catch more). Precision decreases (some false positives). **Strategy 3: Domain-Specific Validation** - Medical facts must be verified against: FDA database, PubMed, Clinical guidelines. ANY medical claim without source → Block. **Strategy 4: Conservative Defaults** - When uncertain: Default to "I don\'t have verified medical information. Consult your doctor." Never guess on medical topics. **Strategy 5: Human Review Queue** - All responses reviewed by medical professional before showing user (for high-risk topics). Or: Show response, flag for review, retract if hallucination found. **Managing False Positives:** Problem: 70% precision = 30% false positives (blocking correct info). Solutions: (1) Phrase carefully: Not "This is wrong" but "I cannot verify this. Please consult medical professional." (2) Provide references: "For verified information, see: [FDA link]". (3) User feedback: "Was this helpful?" to identify false positives. **Implementation:** def medical_response_validation(response): checks_failed = []. if confidence_score < 0.5: checks_failed.append("low_confidence"). if not has_medical_source(response): checks_failed.append("no_source"). if not verified_against_knowledge_base(response): checks_failed.append("not_verified"). if checks_failed: return "I cannot provide unverified medical information. Please consult a healthcare professional." return response. **Result:** Recall: 85% → 98% (only 2% false negatives). Precision: 90% → 65% (35% false positives). Acceptable for medical: Better to over-block than miss hallucinations.',
    keyPoints: [
      'Lower thresholds to increase recall',
      'Combine multiple detection methods',
      'Use domain-specific knowledge bases',
      'Conservative defaults: "I don\'t know" better than hallucination',
    ],
  },
];
