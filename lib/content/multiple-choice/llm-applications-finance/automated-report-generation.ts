export const automatedReportGenerationMultipleChoice = {
  title: 'Automated Report Generation - Multiple Choice',
  id: 'automated-report-generation-mc',
  sectionId: 'automated-report-generation',
  questions: [
    {
      id: 1,
      question:
        'What is the optimal approach for balancing standardization and personalization in automated portfolio reports?',
      options: [
        'Fully standardized reports for all clients to ensure compliance',
        'Fully personalized reports with no standardization',
        'Maintain standard analytical framework and metrics, but customize narrative, focus areas, and language complexity to individual client profiles',
        'Randomize content to appear personalized',
      ],
      correctAnswer: 2,
      explanation:
        'The best approach combines standardized analysis (ensuring consistency, compliance, and quality) with personalized narrative. LLMs enable "mass personalization" where the same analytical framework is applied consistently, but explanations are tailored to client sophistication, examples reference their holdings, and focus aligns with their goals. This provides compliance safety with engagement benefits. Fully standardized (option 0) lacks engagement, fully personalized (option 1) risks compliance issues, and randomization (option 3) is meaningless.',
    },
    {
      id: 2,
      question:
        'When generating reports for periods of poor performance, what is the appropriate tone and approach?',
      options: [
        'Minimize losses and focus only on positive aspects',
        'Use alarming language to emphasize the seriousness',
        'Provide factual, honest assessment of losses with context, explanation, and next steps, avoiding both minimization and panic-inducing language',
        'Blame external factors exclusively',
      ],
      correctAnswer: 2,
      explanation:
        "Ethical and regulatory requirements demand honesty about losses while avoiding panic. Proper approach: clearly state actual performance vs benchmark, provide factual explanation including market context, maintain consistent methodology, and explain actions being taken. Minimizing (option 0) violates trust and regulations, alarming language (option 1) induces panic, and external blame (option 3) lacks accountability. Balance transparency with measured tone, respecting client's need to understand risks.",
    },
    {
      id: 3,
      question:
        'What is the most effective mitigation strategy against LLM hallucination in financial reporting?',
      options: [
        'Using larger, more powerful LLM models',
        'Generating all financial metrics within the LLM for consistency',
        'LLMs generate narrative around pre-calculated, validated numbers rather than calculating during generation, with automated verification of key facts',
        'Accepting hallucination as unavoidable',
      ],
      correctAnswer: 2,
      explanation:
        "The best mitigation separates concerns: financial calculations happen in deterministic systems (verified data pipelines), LLMs generate natural language explanations of that pre-validated data, and automated checks verify quantitative claims against source data. Never allow LLMs to be sole source of truth for financial metrics. Additional: cross-reference all numbers, implement automated sanity checks (percentages sum to 100%, returns calculated correctly), and maintain human review for material facts. Larger models (option 0) don't eliminate hallucination, generating in LLM (option 1) increases risk.",
    },
    {
      id: 4,
      question:
        'What element is most important for maintaining regulatory compliance in automated report generation?',
      options: [
        'Using the longest possible reports with maximum detail',
        'Avoiding any mention of risks or negative performance',
        'Maintaining audit trails of what was generated, ensuring accuracy of disclosures, and human review of material recommendations',
        'Generating reports as quickly as possible',
      ],
      correctAnswer: 2,
      explanation:
        "Regulatory compliance requires: accurate disclosures meeting regulatory requirements, audit trails showing what was generated and when, verification that reports don't contain false information or unsuitable recommendations, and human oversight for material facts and recommendations. Documentation proving data sources and generation process is essential. Length (option 0) doesn't ensure compliance, avoiding risk discussion (option 1) violates requirements, and speed (option 3) without accuracy is dangerous.",
    },
    {
      id: 5,
      question:
        'When personalizing investment reports for different client sophistication levels, what approach best balances accessibility with informativeness?',
      options: [
        'Use identical technical language for all clients',
        'Dumb down all reports to the lowest common denominator',
        'Adjust explanation complexity and terminology to client knowledge level while maintaining substantive content, using LLMs to rephrase concepts appropriately',
        'Omit complex topics entirely for less sophisticated clients',
      ],
      correctAnswer: 2,
      explanation:
        'LLMs enable adjusting explanation complexity without losing substance. For sophisticated clients, use precise technical terms and assume knowledge; for newer investors, explain concepts in accessible language with examples. Same information, different presentation. This respects client needs while ensuring they understand their investments. Identical language (option 0) frustrates some while confusing others, dumbing down (option 1) disrespects sophisticated clients, and omitting topics (option 3) withholds material information.',
    },
  ],
};
