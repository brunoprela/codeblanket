import { MultipleChoiceQuestion } from '../../../types';

export const buildingConversationalAiMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-bcai-mc-1',
      question:
        'When should you summarize a conversation in a conversational AI system?',
      options: [
        'Never summarize - always keep full history',
        'After 50+ messages, summarize old messages, keep last 20 in full',
        'Summarize after every message',
        'Only when user requests',
      ],
      correctAnswer: 1,
      explanation:
        'Optimal summarization strategy: (1) Keep last 20 messages in full (recent context critical), (2) After 50 messages, summarize messages 21-end with Claude Haiku ("Summarize key topics, decisions, context"), (3) Cache summary for 1hr, (4) Include summary in context window (saves 95% tokens: 500k tokens → 2k summary). This balances context quality with cost ($1.50 → $0.08 per conversation).',
    },
    {
      id: 'bcap-bcai-mc-2',
      question:
        'What is the most cost-effective way to create AI character personalities?',
      options: [
        'Fine-tune a model for each character',
        'System prompt + 3-5 few-shot examples per character',
        'Use a single generic prompt for all characters',
        'Hire voice actors',
      ],
      correctAnswer: 1,
      explanation:
        'System prompt + few-shot is optimal: (1) System prompt: Character description, tone, rules (free, instant updates), (2) Few-shot: 3-5 example conversations showing style (500 tokens, strong signal), (3) Periodic reinforcement every 10 messages. Cost: $0 vs $500-2000 fine-tuning. Achieves 85-90% consistency vs 95% fine-tuned, acceptable for most use cases. Fine-tune only for flagship characters.',
    },
    {
      id: 'bcap-bcai-mc-3',
      question:
        'How should PII (Personally Identifiable Information) be handled in chat inputs?',
      options: [
        'Store all PII as-is',
        'Detect with regex, automatically redact ([EMAIL], [SSN]), never send to LLM',
        'Ignore PII concerns',
        'Block all messages with any numbers',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-layer PII protection: (1) Detect: Regex for emails, SSNs, credit cards, phone numbers, (2) Redact automatically: "user@email.com" → "[EMAIL_REDACTED]", (3) Don\'t send PII to LLM (privacy + cost - no need to process), (4) Log redactions (audit trail), (5) User education: "Don\'t share sensitive info." This prevents PII leakage while maintaining functionality.',
    },
    {
      id: 'bcap-bcai-mc-4',
      question: 'What should happen when a jailbreak attempt is detected?',
      options: [
        'Allow it - users can do anything',
        'Block with tiered response: Minor → warning, Major → refusal, log all for analysis',
        'Ban user immediately',
        'Ignore completely',
      ],
      correctAnswer: 1,
      explanation:
        'Balanced jailbreak handling: (1) Detect: LLM classifier trained on known jailbreaks (role-play attacks, hypothetical scenarios, encoded prompts), confidence >0.85 = block, (2) Tiered response: Minor violations → warning + modified response, Major → "I can\'t help with that" + reason, (3) Log all attempts (identify patterns, improve filters), (4) False positive handling: User can report over-filtering, escalate to review. Balance safety and UX.',
    },
    {
      id: 'bcap-bcai-mc-5',
      question:
        'How should conversation state be stored for multiple concurrent conversations per user?',
      options: [
        'Single conversation only',
        'Redis for active (last 10 messages), PostgreSQL for full history, separate by conversation_id',
        'Only in browser localStorage',
        'Everything in a single database table',
      ],
      correctAnswer: 1,
      explanation:
        'Hybrid storage: (1) Redis: Cache active conversation (last 10 messages), fast access during chat, (2) PostgreSQL: Full history (all messages), indexed by conversation_id, (3) Lazy loading: Load older messages on demand, (4) Auto-archive: Move inactive (>7 days) to cold storage (S3), (5) Each conversation separate (by ID), share user profile context across all. This balances performance and cost.',
    },
  ];
