/**
 * Quiz questions for Negative Prompting & Constraints section
 */

export const negativepromptingconstraintsQuiz = [
  {
    id: 'peo-negative-q-1',
    question:
      'Why are "do NOT" instructions sometimes more effective than "do" instructions? Provide examples where negative prompts prevent specific failure modes.',
    hint: 'Think about boundary setting, hallucination prevention, and common LLM mistakes.',
    sampleAnswer:
      'Negative instructions set boundaries and prevent specific bad behaviors: 1) HALLUCINATION PREVENTION: "Do not make up facts or URLs" explicitly blocks fabrication; positive version "be accurate" too vague; 2) SCOPE CONTROL: "Do not discuss politics" clearer than "focus on technical topics" which doesn\'t exclude politics; 3) FORMAT VIOLATIONS: "Do not add explanatory text outside JSON" prevents common wrapping behavior; 4) SAFETY: "Do not execute code" explicit boundary vs "describe code" which might be interpreted flexibly. EXAMPLES: Code generation - "Do not include test code" vs "write only implementation" (second ambiguous); Customer support - "Do not apologize excessively" vs "be direct" (first prevents over-apologizing pattern); Data extraction - "Do not include NULL for missing values" vs "return valid data only" (first handles edge case explicitly). Negative prompts work because they address specific observed failure modes rather than hoping positive instruction prevents them implicitly.',
    keyPoints: [
      'Negative instructions set explicit boundaries',
      'Prevent specific observed failure modes',
      'More effective for hallucination prevention',
      'Clearer for scope and format control',
      'Address LLM tendencies directly',
      'Complement positive instructions',
    ],
  },
  {
    id: 'peo-negative-q-2',
    question:
      'Design a comprehensive constraint system for a customer support AI. Include behavior, content, scope, and safety constraints.',
    hint: 'Think about tone, information boundaries, escalation rules, and privacy.',
    sampleAnswer:
      'Customer support AI constraints: BEHAVIOR: "Do not apologize more than once per response. Do not use phrases like \'I understand your frustration\' repeatedly. Be solution-focused, not empathetic to excess. Do not provide generic responses." CONTENT: "Do not discuss competitor products. Do not make promises about future features. Do not share internal company information. Do not quote pricing without checking current rates." SCOPE: "Only handle: account issues, billing questions, basic troubleshooting. Do NOT: provide legal advice, make refund decisions >$100, discuss medical/health topics, access user accounts directly." SAFETY: "Do not request: passwords, full credit card numbers, SSN. Do not store: personally identifiable information in conversation logs. Always include: privacy disclaimer, data handling notice." FORMAT: "Keep responses under 150 words. Use bullet points for multiple steps. Always end with clear next action." ESCALATION: "Escalate to human if: user frustrated after 3 exchanges, security concerns, request outside scope, unable to resolve in 5 minutes." This comprehensive system ensures safe, effective, bounded support.',
    keyPoints: [
      'Behavior: control tone and style',
      'Content: set information boundaries',
      'Scope: define capabilities clearly',
      'Safety: protect user privacy',
      'Format: ensure consistency',
      'Escalation: define human handoff rules',
    ],
  },
  {
    id: 'peo-negative-q-3',
    question:
      'How do you test the effectiveness of constraints? What metrics indicate whether constraints are being followed?',
    hint: 'Consider violation detection, test cases, automated checking, and monitoring.',
    sampleAnswer:
      'Testing constraints: 1) ADVERSARIAL TEST CASES: Create inputs designed to trigger violations - test if "do not apologize" prevents "I\'m sorry" responses, if scope limits work by asking out-of-scope questions; 2) AUTOMATED DETECTION: Parse outputs for constraint violations - regex for banned phrases, length checks for format constraints, topic classifier for scope violations; 3) METRICS: Violation rate (% outputs breaking constraints), severity scoring (minor vs major violations), constraint adherence score; 4) HUMAN REVIEW: Sample outputs (100-200), annotate violations, calculate inter-rater agreement; 5) A/B TESTING: Compare constrained vs unconstrained variants on violation rates and task success. MONITORING IN PRODUCTION: Real-time violation detection, alert on rates >5%, log violations for analysis, periodic human audit. EXAMPLE: "Do not make up facts" constraint → test with factual questions → measure hallucination rate → should drop from 15% to <3%. If constraint ineffective (still high violations), strengthen wording or add examples.',
    keyPoints: [
      'Create adversarial test cases to trigger violations',
      'Automate detection with parsing and classification',
      'Track violation rate and severity',
      'Use human review for quality assessment',
      'A/B test constrained vs unconstrained',
      'Monitor violations in production continuously',
    ],
  },
];
