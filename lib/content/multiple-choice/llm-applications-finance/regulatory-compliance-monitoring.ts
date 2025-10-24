export const regulatoryComplianceMonitoringMultipleChoice = {
  title: 'Regulatory Compliance & Monitoring - Multiple Choice',
  id: 'regulatory-compliance-monitoring-mc',
  sectionId: 'regulatory-compliance-monitoring',
  questions: [
    {
      id: 1,
      question:
        'Who bears ultimate responsibility when an LLM-based compliance system fails to flag a regulatory violation?',
      options: [
        'The AI vendor that provided the LLM',
        'The LLM model itself',
        'The firm and its compliance officers, regardless of tools used',
        'No one, because AI made the decision',
      ],
      correctAnswer: 2,
      explanation:
        'Ultimate responsibility remains with the firm and compliance officers regardless of tools used. Regulators expect human judgment in compliance, "the AI said it was fine" is not an acceptable defense, firms must demonstrate they understood and validated AI decisions, and they face same penalties as if humans missed the violation. Firms are responsible for: choosing appropriate tools, validating AI before deployment, ongoing monitoring, maintaining oversight, and investigating failures. AI is a tool that amplifies judgment, doesn\'t replace accountability.',
    },
    {
      id: 2,
      question:
        'How should compliance monitoring systems optimally balance sensitivity (catching violations) vs specificity (avoiding false positives)?',
      options: [
        'Always maximize sensitivity regardless of false positives',
        'Always maximize specificity to eliminate false positives',
        'Use tiered alerts and dynamic thresholds based on violation severity, investigation capacity, and regulatory scrutiny level',
        'Use completely random thresholds',
      ],
      correctAnswer: 2,
      explanation:
        'Optimal balance requires: tiered severity levels (high/medium/low), thresholds adjusted by risk type (market manipulation needs high sensitivity even with many false positives; minor disclosure errors can tolerate lower sensitivity), consideration of investigation capacity, and dynamic adjustment during high-risk periods. Track both false positive (efficiency) and false negative (effectiveness) rates. Different risks need different thresholds. Pure sensitivity (option 0) causes alert fatigue, pure specificity (option 1) misses real violations.',
    },
    {
      id: 3,
      question:
        'What is regulatory arbitrage in the context of LLM compliance systems, and why is it dangerous?',
      options: [
        'Using multiple LLM providers for redundancy',
        'LLMs learning to technically comply with rule text while violating regulatory intent through loophole exploitation',
        'Testing compliance systems in multiple jurisdictions',
        'Comparing LLM performance to human compliance officers',
      ],
      correctAnswer: 1,
      explanation:
        'Regulatory arbitrage is using LLMs to identify loopholes that satisfy letter of law while violating spirit—automated adversarial compliance. Examples: finding magic words that make prohibited communication "compliant," structuring transactions to exploit gaps, generating disclosure that meets requirements but obscures information. Dangerous because: regulators focus on substance over form, intentional gaming treated as worse than simple violations, sophisticated evasion suggests deliberate behavior, and reputational damage. Prevention: train on regulatory intent not just text, implement "reasonable person" test, maintain conservative interpretation bias.',
    },
    {
      id: 4,
      question:
        'What is the most effective approach for monitoring employee communications for compliance violations at scale?',
      options: [
        'Reading every message manually',
        'Not monitoring any communications',
        'Using LLMs to flag suspicious patterns, policy violations, and unusual language while maintaining human review of flagged items and privacy considerations',
        'Automatically blocking all employee communications',
      ],
      correctAnswer: 2,
      explanation:
        'Effective monitoring combines LLM capabilities with human judgment: LLMs flag suspicious patterns (sudden deletion patterns, encryption use, unusual recipient patterns), detect policy violations (insider information sharing, market manipulation language), identify tone shifts or evasion, and handle scale. Human compliance officers review flags, make final determinations, and handle nuanced situations. Include privacy protections and clear policies. Manual reading (option 0) is impossible at scale, no monitoring (option 1) violates requirements, blocking all (option 3) prevents legitimate work.',
    },
    {
      id: 5,
      question:
        'What is the most important consideration when implementing LLM-based regulatory change monitoring?',
      options: [
        'Only monitoring regulations in one jurisdiction',
        'Assuming LLMs perfectly understand all regulatory changes',
        'Combining LLM analysis of regulatory updates with expert human interpretation, impact assessment, and implementation planning',
        'Ignoring regulatory changes that seem minor',
      ],
      correctAnswer: 2,
      explanation:
        "Effective regulatory monitoring combines LLM capabilities (tracking multiple jurisdictions, detecting changes, extracting key points) with human expertise (interpreting implications, assessing business impact, planning implementation, understanding context). LLMs process volume and speed, humans provide judgment and accountability. LLMs don't perfectly understand (option 1), single jurisdiction (option 0) misses important changes, and minor changes (option 3) can have major implications. Hybrid approach leverages strengths of both.",
    },
  ],
};
