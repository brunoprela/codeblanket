export const riskAssessmentLLMsMultipleChoice = {
  title: 'Risk Assessment with LLMs - Multiple Choice',
  id: 'risk-assessment-llms-mc',
  sectionId: 'risk-assessment-llms',
  questions: [
    {
      id: 1,
      question:
        'What type of "soft" credit risk signal can LLMs detect that traditional quantitative models typically miss?',
      options: [
        'Current debt-to-equity ratios',
        'Management tone shifts from confident to defensive across filings, increased hedging language, and conflicts between narrative and numbers',
        'Historical stock price movements',
        'Current interest rates',
      ],
      correctAnswer: 1,
      explanation:
        'LLMs excel at detecting qualitative early warning signs in text: management tone shifting from confident to defensive, increased use of hedging language and caveats, vague or evasive explanations around operations, and conflicts between optimistic narrative and declining metrics. These linguistic signals often appear before problems show up in financial ratios. Traditional quantitative models use financial ratios (option 0), stock prices (option 2), and interest rates (option 3), which are lagging indicators.',
    },
    {
      id: 2,
      question:
        'When using LLMs for geopolitical risk assessment, what is the most realistic expectation of LLM capabilities?',
      options: [
        'LLMs can accurately predict geopolitical events before they occur',
        'LLMs are useless for geopolitical analysis',
        'LLMs can rapidly assess exposure and impact once events occur, and synthesize diverse information, but cannot reliably predict black swan events',
        'LLMs should be used exclusively for geopolitical prediction',
      ],
      correctAnswer: 2,
      explanation:
        'Realistic expectation: LLMs excel at rapid impact assessment once events occur, identifying which portfolio companies are affected, synthesizing diverse information sources, and learning from historical similar events. However, they cannot reliably predict unpredictable events (black swans). Best use is proactive scenario analysis for known risks combined with rapid response systems for actual events, recognizing some risks are inherently unpredictable. Claiming prediction capability (option 0) is unrealistic, while dismissing entirely (option 1) wastes valuable tool.',
    },
    {
      id: 3,
      question:
        'How should early warning systems for financial risk optimally balance sensitivity (catching problems) vs specificity (avoiding false alarms)?',
      options: [
        'Maximize sensitivity regardless of false positives',
        'Maximize specificity to eliminate all false positives',
        'Use tiered alert system with different thresholds based on risk severity, portfolio concentration, and investigation capacity, tracking both false positive and false negative rates',
        'Use the same threshold for all types of risks',
      ],
      correctAnswer: 2,
      explanation:
        'Optimal balance requires: tiered severity (high/medium/low), thresholds adjusted for risk type (liquidity risk more sensitive than long-term solvency), consideration of portfolio concentration (key holdings need tighter monitoring), and capacity awareness (team can handle 10 alerts/day, not 100). Track both false positive rate (efficiency) and false negative rate (effectiveness). Different risks need different thresholds. Maximum sensitivity (option 0) creates alert fatigue, maximum specificity (option 1) misses real problems.',
    },
    {
      id: 4,
      question:
        'What is the most significant challenge when using LLMs to assess counterparty risk?',
      options: [
        'LLMs cannot read financial statements',
        'Timeliness and completeness of available information about private counterparties, plus difficulty assessing interconnection risks',
        'LLMs are biased toward approving all counterparties',
        'Counterparty risk assessment is too simple for LLMs',
      ],
      correctAnswer: 1,
      explanation:
        "Major challenges include: private companies have limited public information (unlike public companies with SEC filings), information may be outdated when available, difficulty assessing interconnection risks and contagion potential, and opacity of counterparty's own counterparties. LLMs can help synthesize available information but are limited by information availability and timeliness. They don't have inherent bias toward approval (option 2), can read statements (option 0), and counterparty risk is complex not simple (option 3).",
    },
    {
      id: 5,
      question:
        'In credit risk analysis, what advantage do LLMs provide over traditional credit scoring models?',
      options: [
        'LLMs eliminate the need for financial data',
        'LLMs are always more accurate than quantitative models',
        'LLMs can incorporate qualitative information from management discussion, industry context, and narrative analysis alongside quantitative metrics for more holistic assessment',
        'LLMs process credit decisions faster',
      ],
      correctAnswer: 2,
      explanation:
        "LLMs complement traditional models by adding qualitative analysis: management discussion tone and changes, industry context and competitive positioning, risk factor analysis, and narrative consistency with numbers. Combined with quantitative credit metrics, this provides more complete picture. LLMs don't eliminate need for data (option 0), aren't always more accurate than proven quantitative models (option 1), and speed (option 3) isn't the primary advantage. The value is incorporating information traditional models cannot process.",
    },
  ],
};
