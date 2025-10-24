/**
 * Quiz questions for Output Validation & Guardrails section
 */

export const outputvalidationguardrailsQuiz = [
  {
    id: 'output-val-q-1',
    question:
      'Your LLM outputs JSON that should match schema: {"name": str, "age": int, "email": str}. 15% of outputs fail validation (wrong types, missing fields). Design a system that reduces validation failures to <2% while maintaining output quality.',
    hint: 'Consider prompting improvements, retry logic, and fallback strategies.',
    sampleAnswer:
      '**Problem:** 15% schema validation failures. Need: <2%. **Root Cause Analysis:** Why failures? (1) LLM generates extra fields. (2) Wrong types: age="30" (string) not 30 (int). (3) Missing required fields. **Solution: Multi-Layer Approach:** **Layer 1: Improved Prompting** - Be explicit about schema: prompt = """Generate JSON matching EXACTLY this schema: {"name": "string", "age": number, "email": "string"}. REQUIRED fields: name, age, email. FORBIDDEN: Do not add extra fields. Example valid output: {"name": "John", "age": 30, "email": "john@example.com"}""". Result: 15% → 8% failures. **Layer 2: Output Formatting Instructions** - Add: "Return ONLY valid JSON. No explanation before or after." Prevents: "Here\'s the JSON: {...}" (invalid). Result: 8% → 5% failures. **Layer 3: Retry with Validation Feedback** - def generate_with_retry(prompt, max_retries=3): for attempt in range(max_retries): output = llm(prompt). if validate_schema(output): return output. else: error = get_validation_error(output). prompt += f"\\\\nPrevious attempt failed: {error}. Try again.". return fallback. Result: 5% → 1.5% failures. **Layer 4: Automatic Repair** - For minor issues, repair automatically: if output["age",] is string: output["age",] = int(output["age",]). if "extra_field" in output: del output["extra_field",]. Result: 1.5% → 0.5% failures. **Layer 5: Fallback Strategies** - If still failing after retries: Return safe default: {"name": "Unknown", "age": 0, "email": "invalid"}. Or: Use structured output API (OpenAI JSON mode, Function calling). Result: <2% failures ✓.',
    keyPoints: [
      'Explicit schema in prompt with examples',
      'Retry with validation feedback',
      'Automatic repair for minor issues',
      'Fallback for persistent failures',
    ],
  },
  {
    id: 'output-val-q-2',
    question:
      'Design a quality validation system that goes beyond schema checking. How do you validate that LLM outputs are relevant, complete, and high-quality? What metrics would you use, and how do you handle low-quality outputs?',
    hint: 'Consider relevance, completeness, coherence, and user feedback.',
    sampleAnswer:
      '**Quality Dimensions:** **1. Relevance** - Does output answer the question? Metric: Semantic similarity between prompt and response. def check_relevance(prompt, response): prompt_embedding = embed(prompt). response_embedding = embed(response). similarity = cosine_similarity(prompt_embedding, response_embedding). return similarity > 0.6  # Threshold. **2. Completeness** - Is response complete or truncated? Check: if response.endswith("...") or "continued" in response: incomplete = True. if len(response) < 50: likely_incomplete = True. **3. Coherence** - Is response internally consistent? Check for: Repeated sentences, Contradictions, Non-sequiturs. def check_coherence(response): sentences = split_sentences(response). if len(sentences) != len(set(sentences)): # Repeated sentences, coherent = False. return coherent. **4. Factual Consistency** - Use LLM to judge: judge_prompt = f"Is this response factually consistent? {response}". consistency_score = llm_judge(judge_prompt). **5. User Intent Match** - Does response match likely user intent? Use classification: intent = classify_intent(prompt)  # question, command, conversation. appropriate = check_if_appropriate_for_intent(response, intent). **Combined Quality Score:** def calculate_quality(prompt, response): scores = {relevance: check_relevance(prompt, response), completeness: check_completeness(response), coherence: check_coherence(response), factual: check_factual_consistency(response)}. overall = sum(scores.values()) / len(scores). return overall, scores. **Handling Low Quality:** if quality_score < 0.6: action = "regenerate". elif quality_score < 0.7: action = "flag_for_review". else: action = "allow". **User Feedback:** Track: "Was this helpful?" If < 60% positive: quality issue. **Result:** Multi-dimensional quality beyond schema.',
    keyPoints: [
      'Check relevance, completeness, coherence',
      'Use embeddings for semantic similarity',
      'LLM-as-judge for quality assessment',
      'Regenerate or flag low-quality outputs',
    ],
  },
  {
    id: 'output-val-q-3',
    question:
      'Your output validation adds 200ms latency. Users complain about slowness. Optimize validation to <50ms while maintaining safety. What checks can be parallelized, cached, or simplified?',
    hint: 'Consider async validation, caching, and prioritizing critical checks.',
    sampleAnswer:
      '**Current: 200ms Validation.** Breakdown: Schema validation: 5ms. PII detection: 80ms. Content moderation: 120ms (OpenAI API). Quality checks: 30ms. Bias detection: 15ms. **Optimization:** **Strategy 1: Parallel Execution** - Run checks simultaneously: async def validate_parallel(output): results = await asyncio.gather(schema_check(output), pii_detect(output), content_moderate(output), quality_check(output), bias_detect(output)). return combine(results). Result: 200ms → 120ms (limited by slowest check). **Strategy 2: Prioritize Critical Checks** - Run critical checks first, skip others if failing: if not schema_valid(output):  # 5ms, return fail  # Skip remaining checks. if has_pii(output):  # 80ms, return fail. # Only if passed critical, run nice-to-have, quality_check(output)  # Async in background. Result: Fast-fail on critical issues. **Strategy 3: Caching** - Cache validation results: cache_key = hash(output). if cache_key in validation_cache: return cached_result  # 2ms. Result: 80% cache hit rate → 0.8 × 2ms + 0.2 × 120ms = 25.6ms. **Strategy 4: Faster Models** - Replace: OpenAI Moderation API (120ms, $0.0002) with: Local toxicity model (40ms, free). Trade-off: Slightly lower accuracy, but 3x faster. **Strategy 5: Async Post-Validation** - Allow output immediately, validate in background: return output_to_user  # 0ms perceived. validate_async(output)  # Background. if violation: retract_within_5s(). Risk: Brief window where unsafe content visible. Acceptable for low-risk apps. **Optimized System:** Critical checks (parallel): 5ms + max(80ms, 40ms, 30ms) = 85ms. With caching (80% hit): 0.8 × 2ms + 0.2 × 85ms = 18.6ms ✓. Result: <50ms target achieved.',
    keyPoints: [
      'Parallel execution reduces total time',
      'Prioritize critical checks, skip non-critical if fail',
      'Caching for repeated validations',
      'Consider async validation for non-critical checks',
    ],
  },
];
