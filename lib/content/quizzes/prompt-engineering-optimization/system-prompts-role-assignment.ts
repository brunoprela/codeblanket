/**
 * Quiz questions for System Prompts & Role Assignment section
 */

export const systempromptsroleassignmentQuiz = [
    {
        id: 'peo-system-q-1',
        question:
            'How do system prompts differ from user prompts in terms of authority and persistence? Explain how this affects production AI application behavior.',
        hint: 'Consider priority, scope, and how models treat different message types.',
        sampleAnswer:
            'System prompts have higher authority than user prompts - they set persistent behavior that applies across the entire conversation while user prompts handle specific tasks. Models treat system prompts as authoritative instructions that define the AI\'s role, capabilities, and boundaries. In production, this means system prompts: 1) Define consistent personality/behavior across all interactions; 2) Set guardrails that user inputs shouldn\'t override; 3) Establish output formats and quality standards; 4) Define scope and limitations. The persistence means you set behavior once rather than repeating instructions in every turn. However, this also means system prompt errors affect all outputs, making testing critical. Production apps like Cursor use system prompts to establish code-editing behavior that persists while user prompts specify individual edit tasks.',
        keyPoints: [
            'System prompts have higher authority',
            'They persist across entire conversation',
            'Define consistent AI behavior',
            'Set boundaries user prompts can\'t override',
            'Critical for production consistency',
            'Errors affect all outputs - test thoroughly',
        ],
    },
    {
        id: 'peo-system-q-2',
        question:
            'Design a system prompt for a code review AI that handles multiple programming languages. What behavioral guidelines, constraints, and output formats would you include?',
        hint: 'Think about role definition, expertise, tone, output structure, and what the AI should and shouldn\'t do.',
        sampleAnswer:
            'SYSTEM PROMPT: "You are a senior code reviewer with expertise in Python, JavaScript, TypeScript, Java, and Go. Your reviews focus on correctness, security, performance, and maintainability. BEHAVIOR: Be constructive and specific - explain WHY changes are needed. Prioritize critical issues (bugs, security) over style preferences. CONSTRAINTS: Only review languages in your expertise. If code has syntax errors, identify them first. Never execute or run code. Don\'t suggest changes without explanation. OUTPUT FORMAT: ## Summary (one sentence) ## Critical Issues (bugs/security with severity HIGH/MEDIUM/LOW) ## Code Quality (maintainability/style) ## Positive Notes (what\'s done well) ## Recommendations (prioritized list). TONE: Professional and educational. Assume developer wants to learn. Avoid being condescending. Focus on teaching principles, not just pointing out issues." This covers role (senior reviewer), expertise (languages), behavior (constructive, prioritized), constraints (safety, scope), format (structured sections), and tone (educational).',
        keyPoints: [
            'Define specific role and expertise',
            'Set behavioral guidelines (tone, priorities)',
            'Include explicit constraints (scope, safety)',
            'Specify structured output format',
            'Define what AI can and cannot do',
            'Consider edge cases and error handling',
        ],
    },
    {
        id: 'peo-system-q-3',
        question:
            'How would you A/B test different system prompts in production? What metrics would you track and how would you determine the winner?',
        hint: 'Consider traffic splitting, statistical significance, multiple metrics, and business impact.',
        sampleAnswer:
            'A/B testing system prompts requires: 1) TRAFFIC SPLITTING: Route 50% of users to variant A, 50% to variant B using consistent hashing (same user always sees same variant); 2) METRICS: Track accuracy/success rate, user satisfaction ratings, task completion rate, error rates, latency, cost per request, and user engagement metrics; 3) DURATION: Run for sufficient time to get statistical significance (typically 1-2 weeks or thousands of requests); 4) STATISTICAL ANALYSIS: Use t-test for continuous metrics, chi-square for categorical. Ensure p-value < 0.05 for significance; 5) BUSINESS IMPACT: Consider not just accuracy but user satisfaction, retention, and business KPIs; 6) DECISION FRAMEWORK: Variant wins if it\'s statistically significantly better on primary metric AND not significantly worse on other metrics. Implementation: Create prompt version registry, implement random assignment with logging, automated metrics collection, dashboard showing real-time results, and ability to quickly end test if variant is clearly worse.',
        keyPoints: [
            'Split traffic consistently (e.g., 50/50)',
            'Track multiple metrics: accuracy, cost, latency, satisfaction',
            'Run for statistical significance',
            'Use proper statistical tests (t-test, chi-square)',
            'Consider business impact, not just accuracy',
            'Implement safety nets and quick rollback',
        ],
    },
];

