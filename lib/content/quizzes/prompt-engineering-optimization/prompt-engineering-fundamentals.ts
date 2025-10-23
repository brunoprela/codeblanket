/**
 * Quiz questions for Prompt Engineering Fundamentals section
 */

export const promptengineeringfundamentalsQuiz = [
  {
    id: 'peo-fundamentals-q-1',
    question:
      'Explain the core components of a well-structured prompt (Role, Task, Constraints, Format, Examples). Why is each component important for production reliability?',
    hint: 'Think about how each component contributes to consistent and predictable outputs at scale.',
    sampleAnswer:
      'A well-structured prompt has five key components: 1) ROLE sets context about who the AI is and its expertise, establishing consistent behavior; 2) TASK clearly defines what needs to be done, eliminating ambiguity; 3) CONSTRAINTS set boundaries and rules, preventing unwanted behaviors; 4) FORMAT specifies output structure, making results parseable and consistent; 5) EXAMPLES demonstrate desired behavior, dramatically improving reliability through few-shot learning. In production, this structure ensures predictable outputs across thousands of requests, reduces error rates, and makes prompts maintainable and versionable like code. Each component addresses a specific failure mode - role prevents inconsistent personality, task prevents misunderstanding, constraints prevent boundary violations, format prevents parsing errors, and examples prevent quality variations.',
    keyPoints: [
      'Role establishes AI identity and expertise',
      'Task provides clear, unambiguous instructions',
      'Constraints set boundaries and prevent issues',
      'Format ensures parseable, consistent outputs',
      'Examples dramatically improve reliability',
      'Structure makes prompts maintainable and testable',
    ],
  },
  {
    id: 'peo-fundamentals-q-2',
    question:
      'When should you use zero-shot versus few-shot prompting? Provide specific scenarios where each approach is optimal.',
    hint: 'Consider task complexity, model capability, consistency requirements, and cost implications.',
    sampleAnswer:
      'Zero-shot is optimal when: 1) Using powerful models (GPT-4) on simple, well-defined tasks where the model already understands the pattern; 2) Task is common enough that the model has seen many examples during training; 3) Flexibility in output format is acceptable; 4) Cost/latency is critical and adding examples would be wasteful. Few-shot is better when: 1) Task is nuanced or domain-specific requiring pattern demonstration; 2) Consistency in output format is critical for parsing; 3) Edge cases need explicit handling through examples; 4) Task is uncommon or requires specific style/tone. In production, few-shot is generally safer as it reduces variance. For instance, use zero-shot for "translate to French" but few-shot for "extract entities in specific JSON format" where structure matters.',
    keyPoints: [
      'Zero-shot works for simple, well-defined tasks',
      'Few-shot essential for consistent format',
      'Few-shot handles nuanced patterns better',
      'Zero-shot saves tokens when appropriate',
      'Few-shot reduces output variance in production',
      'Choose based on task complexity and consistency needs',
    ],
  },
  {
    id: 'peo-fundamentals-q-3',
    question:
      'How would you build a systematic testing framework for prompts? What metrics would you track and how would you compare different prompt versions?',
    hint: 'Think about test cases, evaluation functions, metrics, and statistical significance.',
    sampleAnswer:
      "A systematic prompt testing framework needs: 1) TEST CASES: Diverse set of inputs covering common cases, edge cases, and known failure modes with expected outputs; 2) EVALUATION FUNCTIONS: Automated scoring (exact match, semantic similarity, format validation) plus human evaluation for quality; 3) METRICS: Track accuracy, consistency (output variance), latency, cost per request, and overall quality score; 4) VERSION COMPARISON: Run A/B tests with statistical significance testing (t-test, confidence intervals), track performance over time, and measure improvement delta; 5) REGRESSION TESTING: Ensure new versions don't break existing functionality; 6) PRODUCTION MONITORING: Real-world performance tracking with alerts on degradation. Implementation should include version control for prompts, automated test runs on every change, dashboards showing metrics over time, and ability to quickly rollback if a new version underperforms.",
    keyPoints: [
      'Comprehensive test cases including edge cases',
      'Automated and human evaluation methods',
      'Track multiple metrics: accuracy, consistency, cost',
      'A/B test with statistical significance',
      'Version control and regression testing',
      'Continuous monitoring in production',
    ],
  },
];
