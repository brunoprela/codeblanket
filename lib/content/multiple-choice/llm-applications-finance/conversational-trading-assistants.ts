export const conversationalTradingAssistantsMultipleChoice = {
  title: 'Conversational Trading Assistants - Multiple Choice',
  id: 'conversational-trading-assistants-mc',
  sectionId: 'conversational-trading-assistants',
  questions: [
    {
      id: 1,
      question:
        'What is the most critical safety mechanism for voice-activated trading systems?',
      options: [
        'Processing commands as quickly as possible',
        'Multi-layer verification including explicit confirmation, numerical/ticker repetition, position size limits, and validation checks before execution',
        'Allowing unlimited position sizes',
        'Never requiring user confirmation',
      ],
      correctAnswer: 1,
      explanation:
        'Critical safety requires multiple layers: explicit confirmation with details repeated back to user ("I heard sell 100 shares AAPL at market, say yes to confirm"), audio quality checks rejecting unclear commands, position size limits requiring additional authentication, context-aware validation (can\'t sell stock you don\'t own), and execution-level checks. This prevents catastrophic errors from misheard commands or unintended activations. Speed without safety (option 0), unlimited sizes (option 2), and no confirmation (option 3) are dangerous.',
    },
    {
      id: 2,
      question:
        'How should conversational trading assistants handle emotionally-charged situations where users might make poor decisions during market volatility?',
      options: [
        'Execute all commands immediately regardless of context',
        'Refuse to execute any trades during volatile periods',
        'Acknowledge emotional state, provide factual context, introduce friction for potentially harmful trades, and escalate severe situations to human advisors',
        'Make decisions on behalf of the user',
      ],
      correctAnswer: 2,
      explanation:
        'Appropriate handling balances user autonomy with duty of care: acknowledge emotional state without condescension, provide factual context (market history, volatility norms), remind of long-term plan, introduce cooling-off periods for potentially harmful emotional trades, and escalate severe situations to human advisors. Should not refuse all trades (option 1), execute all without consideration (option 0), or make decisions for users (option 3). Goal is protecting users from impulsive harmful decisions while respecting autonomy.',
    },
    {
      id: 3,
      question:
        'What is the primary regulatory compliance requirement for automated trading assistants?',
      options: [
        'No specific compliance requirements exist',
        'Only speed of execution matters',
        'Maintaining complete audit trails linking conversations to executions, ensuring suitability, and handling disputes with documented evidence',
        'Allowing all trades the user requests',
      ],
      correctAnswer: 2,
      explanation:
        "Regulatory compliance requires: complete audio/text logging of all interactions, transcription with confidence scores, record of parsed intent and validation, portfolio state at trade time, biometric verification, suitability verification (trade consistent with risk profile and objectives), and documentation for dispute resolution. Systems must prove authorization, demonstrate controls preventing unsuitable trades, and maintain audit trails. Compliance requirements definitely exist (option 0), speed isn't primary concern (option 1), and allowing all trades (option 3) violates suitability requirements.",
    },
    {
      id: 4,
      question:
        'When parsing natural language trading commands, what indicates the highest-quality implementation?',
      options: [
        'Executing commands without confirmation',
        'Only accepting precisely formatted commands',
        'Parsing intent, identifying ambiguities, performing safety checks, providing clear confirmations, and escalating unclear commands for clarification',
        'Guessing user intent when commands are unclear',
      ],
      correctAnswer: 2,
      explanation:
        'High-quality systems: parse intent accurately, identify ambiguities or unclear aspects, perform comprehensive safety checks (valid ticker, sufficient funds, reasonable quantity, price in range), generate clear confirmation messages, ask clarifying questions when needed, and escalate borderline cases. This prevents errors while maintaining usability. No confirmation (option 0) is dangerous, requiring precise formats (option 1) defeats conversational purpose, and guessing (option 3) leads to errors.',
    },
    {
      id: 5,
      question:
        'What is the appropriate role for LLMs in explaining portfolio holdings to users?',
      options: [
        'LLMs should not explain holdings at all',
        'LLMs should only provide technical financial jargon',
        'LLMs should generate clear, conversational explanations of holdings, performance, context, and suitability in language appropriate to user sophistication',
        'LLMs should make buy/sell recommendations without user input',
      ],
      correctAnswer: 2,
      explanation:
        'LLMs excel at explaining holdings in accessible language: what the company does, how position performed, why it might be performing that way, whether position size is appropriate, and contextualizing within portfolio and goals. Adjust complexity to user knowledge level. This builds understanding and engagement. Not explaining (option 0) wastes opportunity, only jargon (option 1) fails many users, and autonomous recommendations (option 3) exceed appropriate role. LLMs should inform and explain, humans decide.',
    },
  ],
};
