/**
 * Quiz questions for Prompt Optimization Techniques section
 */

export const promptoptimizationtechniquesQuiz = [
  {
    id: 'peo-optimization-q-1',
    question:
      'Describe a systematic process for iteratively optimizing a prompt. What data do you collect and how do you measure improvement?',
    hint: 'Think about test cases, failure analysis, metrics, and iteration cycles.',
    sampleAnswer:
      'Systematic optimization process: 1) BASELINE: Start with simple prompt, run on comprehensive test set, measure performance (accuracy, latency, cost); 2) FAILURE ANALYSIS: Identify patterns in failures - what inputs fail most, what errors occur, why; 3) HYPOTHESIS GENERATION: Based on failures, hypothesize improvements (add examples for X cases, clarify format for Y cases, add constraints for Z); 4) VARIANT CREATION: Create improved prompts addressing failure modes; 5) A/B TESTING: Run variants vs baseline with statistical rigor; 6) SELECTION: Choose variant with significant improvement; 7) ITERATE: Repeat on new failures. METRICS: Track accuracy/success rate (primary), consistency (output variance), latency, cost per request, edge case handling. STOPPING CRITERIA: Reach 95%+ accuracy, diminishing returns, or cost/benefit optimal. TOOLS: Version control for prompts, automated test runner, metrics dashboard, regression test suite. Expect 3-5 iterations to reach production quality.',
    keyPoints: [
      'Start with baseline measurements',
      'Analyze failure patterns systematically',
      'Generate hypotheses from failures',
      'A/B test variants with statistics',
      'Track multiple metrics: accuracy, cost, latency',
      'Iterate until diminishing returns',
    ],
  },
  {
    id: 'peo-optimization-q-2',
    question:
      'What is prompt compression and when should you use it? Explain techniques like LLMLingua and their trade-offs.',
    hint: 'Consider token costs, semantic preservation, compression ratios, and quality impact.',
    sampleAnswer:
      'Prompt compression reduces token count while preserving meaning, saving costs. TECHNIQUES: 1) Basic: Remove redundancy, shorten phrases, eliminate filler words - achieves 10-20% reduction; 2) Structural: Reduce examples, truncate context, compress instructions - 30-40% reduction; 3) LLMLingua: Uses small model to identify and remove unimportant tokens - up to 20x compression with minimal quality loss. WHEN TO USE: Long context windows expensive (many tokens), repetitive information, background context less critical, cost optimization priority. TRADE-OFFS: Risk losing important information, semantic meaning harder to preserve at high compression, adds complexity and latency (LLMLingua), quality-cost balance required. IN PRACTICE: Test compressed vs original on holdout set, measure quality degradation, calculate cost savings, use if quality drop <5% but cost savings >30%. Essential for production systems processing millions of requests. Cursor compresses large codebases this way.',
    keyPoints: [
      'Reduces tokens while preserving meaning',
      'Basic techniques: remove redundancy, shorten phrases',
      'LLMLingua: intelligent compression up to 20x',
      'Use when cost optimization critical',
      'Risk of quality degradation',
      'Test quality impact before deploying',
    ],
  },
  {
    id: 'peo-optimization-q-3',
    question:
      'How do you balance cost vs quality when optimizing prompts? What framework would you use to make this decision?',
    hint: 'Think about business value, error costs, volume, and optimization strategies.',
    sampleAnswer:
      "Cost-quality framework: 1) CALCULATE VALUE: What's business value of accuracy improvement? What's cost of errors? For customer support, one error might cost $10 in customer satisfaction; for medical advice, error catastrophic; 2) MEASURE COSTS: Calculate current cost per request (tokens × price), cost at different quality levels (GPT-4 vs GPT-3.5, longer vs shorter prompts); 3) VOLUME ANALYSIS: High-volume tasks justify optimization effort and cheaper models; low-volume can use expensive models if quality matters; 4) DECISION MATRIX: Plot quality vs cost for different approaches, identify Pareto frontier (best quality for each cost level), choose based on business constraints. STRATEGIES: Use expensive models/prompts for high-value tasks, cheap for high-volume low-stakes; compress prompts for common tasks; cache results for repeated queries; use cascade (try cheap first, expensive if fails). EXAMPLE: Customer email analysis - simple sentiment (GPT-3.5, $0.002) acceptable; complex complaint resolution needs GPT-4 with examples ($0.03) but worth cost.",
    keyPoints: [
      'Calculate business value and error costs',
      'Measure cost at different quality levels',
      'Consider volume - high volume justifies optimization',
      'Plot quality vs cost Pareto frontier',
      'Use expensive models only for high-value tasks',
      'Implement cascading or caching strategies',
    ],
  },
];
