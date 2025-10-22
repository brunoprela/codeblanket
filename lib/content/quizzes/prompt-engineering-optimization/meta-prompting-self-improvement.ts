/**
 * Quiz questions for Meta-Prompting & Self-Improvement section
 */

export const metapromptingselfimprovementQuiz = [
    {
        id: 'peo-meta-q-1',
        question:
            'What is meta-prompting and how is it different from regular prompting? Provide examples of when LLM-generated prompts outperform human-written ones.',
        hint: 'Think about prompt generation, optimization, and what LLMs know about effective prompting.',
        sampleAnswer:
            'Meta-prompting uses LLMs to write/improve prompts for other LLMs. REGULAR vs META: Regular - human writes "Classify sentiment", tests, iterates manually; Meta - human describes task, LLM generates optimal prompt with examples, format specs, constraints automatically. LLM-GENERATED BETTER WHEN: 1) Complex tasks requiring detailed instructions - LLM knows effective patterns from training; 2) Domain-specific tasks where LLM can generate appropriate examples; 3) Format specification - LLM produces cleaner schemas than humans; 4) Constraint definition - comprehensive coverage of edge cases. EXAMPLES: Task "extract entities from medical text" - human might write basic prompt, LLM generates one with medical terminology examples, HIPAA constraints, structured format, error handling. Code generation tasks - LLM can generate prompts with language-specific best practices, comprehensive examples. LIMITATIONS: LLMs can\'t judge business priorities, domain expertise, or measure real-world performance without testing. Best approach: LLM generates, human validates, iterate.',
        keyPoints: [
            'Meta-prompting: LLMs write prompts for LLMs',
            'Generates comprehensive instructions automatically',
            'LLMs know effective prompting patterns',
            'Better at detailed specifications and examples',
            'Can\'t judge business priorities alone',
            'Best: LLM generates, human validates',
        ],
    },
    {
        id: 'peo-meta-q-2',
        question:
            'Design a self-improving prompt system that learns from failures. How does it identify issues and generate improvements automatically?',
        hint: 'Consider failure logging, pattern analysis, improvement generation, and testing loop.',
        sampleAnswer:
            'Self-improving system: 1) FAILURE LOGGING: Track all requests, label failures (wrong answer, format error, edge case miss), store input-expected-actual triples with context; 2) PATTERN ANALYSIS: Aggregate failures weekly, cluster similar failures (e.g., all negative sentiment misclassified as neutral), identify root causes through LLM analysis: "Why did these 20 inputs fail?"; 3) IMPROVEMENT GENERATION: Feed current prompt + failure examples to LLM: "Analyze why prompt failed on these inputs, suggest specific improvements", get 3-5 improvement suggestions; 4) VARIANT CREATION: Generate improved prompts addressing each failure pattern; 5) AUTOMATED TESTING: Run variants on held-out test set + recent failures, measure improvement; 6) SELECTION: Deploy variant with best improvement if statistically significant (p<0.05); 7) MONITORING: Track new performance, rollback if regression. EXAMPLE: Sentiment classifier failing on sarcasm → system detects pattern → generates improvement "explicitly handle sarcastic tone" → tests → deploys. Runs weekly or when failure rate >threshold. Human approves major changes.',
        keyPoints: [
            'Log all failures with full context',
            'Analyze patterns in failures weekly',
            'Use LLM to identify root causes',
            'Generate improvement variants automatically',
            'Test variants on holdout set',
            'Deploy best variant with human approval',
        ],
    },
    {
        id: 'peo-meta-q-3',
        question:
            'What is prompt evolution? How does it differ from gradient descent and why might it find better solutions?',
        hint: 'Think about genetic algorithms, discrete vs continuous optimization, and local minima.',
        sampleAnswer:
            'Prompt evolution applies genetic algorithms to prompts: Generate population of prompts → Evaluate fitness (accuracy on test cases) → Select best → Create mutations (variations) → Repeat. DIFFERS FROM GRADIENT DESCENT: Gradient descent (used in model training) requires continuous parameters and gradients - won\'t work for discrete text prompts. Evolution works on any fitness function without gradients. PROCESS: Start with seed prompt → Generate 5 variations (mutation: "rephrase this", "add examples", "change structure") → Test all on benchmark → Keep top 2 → Generate 5 more from those → Iterate 10 generations. WHY BETTER SOLUTIONS: 1) Explores diverse prompt space (not stuck in local optima); 2) No assumptions about smooth optimization landscape; 3) Can combine features from different prompts; 4) Handles non-differentiable objectives (human preference). LIMITATIONS: Slower than targeted improvement (requires many evaluations), expensive (many LLM calls), might converge to local maximum. WHEN TO USE: Hard optimization problems, have compute budget, want to explore space comprehensively. Combine with human insights for best results.',
        keyPoints: [
            'Evolution: generate variants, test, select, repeat',
            'No gradients needed - works on discrete text',
            'Explores diverse prompt space',
            'Avoids local optima through random mutations',
            'Expensive but finds non-obvious solutions',
            'Combine with targeted improvements',
        ],
    },
];

