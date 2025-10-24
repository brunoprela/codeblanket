export const buildingConversationalAiQuiz = [
  {
    id: 'bcap-bcai-q-1',
    question:
      'Design a conversation memory system that handles: (1) 1000+ message conversations, (2) 200k token context limit, (3) Multiple concurrent conversations per user. How do you decide: which messages to include, when to summarize, how to store efficiently? Compare: storing full history vs summaries vs sliding window. Include cost analysis.',
    sampleAnswer:
      'Hybrid approach: Recent messages (full) + Summary (old messages) + Semantic search (relevant context). Storage: PostgreSQL for messages + Redis for active conversations. Strategy: (1) Always include: last 20 messages (recent context), system prompt, user profile. (2) If conversation >50 messages: Summarize messages 21-end with Claude Haiku: "Summarize this conversation focusing on: key topics, decisions, context needed for future messages." Cache summary for 1hr. (3) Context selection: Allocate 200k tokens: System prompt (1k) + Summary (5k) + Recent messages (est. 30k) + Response buffer (4k) = 40k used. If user asks about older topic: Semantic search over all messages (embed query, vector search), include relevant messages (10k tokens). (4) Cost: Full storage (1000 messages × 500 tokens avg = 500k tokens) → $1.50 to include each time. Summary approach: (1000 messages → 2k token summary) → $0.006 per conversation. Sliding window (last 20 only): Free but loses context. Recommendation: Hybrid saves 95% on tokens ($0.08 vs $1.50 per conversation), maintains quality. Concurrent conversations: User has 5 active conversations. Store separately with IDs, share user profile context across all. Use Redis to cache active conversation state (last 10 messages), lazy load older messages from PostgreSQL. Auto-archive conversations inactive >7 days (move to cold storage).',
    keyPoints: [
      'Hybrid: full recent messages + summary of old + semantic search for relevance',
      'Summarize after 50 messages, cache summary, regenerate every 20 messages',
      'Token allocation: system (1k) + summary (5k) + recent (30k) + response (4k)',
      'Cost: hybrid approach 95% cheaper than full context ($0.08 vs $1.50)',
      'Redis for active conversations, PostgreSQL for history, vector search for retrieval',
    ],
  },
  {
    id: 'bcap-bcai-q-2',
    question:
      'Your conversational AI needs personality. Compare: (1) System prompt engineering, (2) Fine-tuning base model, (3) Few-shot examples, (4) Retrieval-augmented personalities. For a product with 10 different AI characters (e.g., "Professional assistant", "Creative writer", "Technical expert"), which approach is most cost-effective and maintainable? How do you ensure personality consistency across conversations?',
    sampleAnswer:
      'System prompt + few-shot examples (most practical). Reasoning: (1) Fine-tuning: Expensive ($500-2000 per character), hard to iterate, model drift. Only viable for 1-2 core characters. (2) System prompts: Free, instant updates, easy A/B testing. Cons: Can drift from personality in long conversations. (3) Few-shot: 3-5 example conversations per character (500 tokens), strong signal for style. (4) RAG: Retrieve personality-relevant responses, good for consistency but slower. Recommended approach: System prompt (character description, tone, rules) + 3-5 few-shot examples + periodic personality reinforcement. Implementation: (1) Character template: "You are {name}, a {role}. Personality: {traits}. Communication style: {style}. Example conversations: {examples}". (2) Store templates in database, load based on conversation character_id. (3) Consistency: Every 10 messages, inject reminder: "Remember, you are {character}". (4) Drift detection: Classify last response for personality traits, if drift detected, reinforce. (5) User feedback: Thumbs up/down with "Out of character" option. Cost: System prompt approach: $0 per character vs $500-2000 fine-tuning. Maintenance: Update prompt (instant) vs retrain model (days). Hybrid: Use fine-tuned model for flagship character (best experience), system prompts for others (cost-effective). Quality: System prompts achieve 85-90% consistency vs 95% fine-tuned, acceptable for most use cases.',
    keyPoints: [
      'System prompt + few-shot examples most cost-effective (free vs $500-2000 fine-tuning)',
      'Few-shot: 3-5 example conversations per character (strong personality signal)',
      'Consistency: periodic reinforcement every 10 messages, drift detection',
      'Fine-tuning only for flagship characters, system prompts for others',
      'System prompts: instant updates, easy A/B testing, 85-90% consistency',
    ],
  },
  {
    id: 'bcap-bcai-q-3',
    question:
      'Design a safety layer for conversational AI that prevents: (1) Harmful content generation, (2) PII leakage, (3) Jailbreak attempts, (4) Hallucinations about facts. How do you balance: safety vs user experience (not over-filtering)? Include: input moderation, output filtering, and post-hoc analysis. What happens when a safety check fails?',
    sampleAnswer:
      'Multi-layer safety: (1) Input moderation (pre-generation). (2) Output filtering (post-generation). (3) Continuous monitoring. Input: (1) OpenAI moderation API (hate, violence, self-harm) - <100ms, blocks 0.1% of prompts. (2) PII detection: Regex for emails, SSNs, credit cards. Redact automatically: "user@email.com" → "[EMAIL]". (3) Jailbreak detection: LLM classifier (Claude Haiku) trained on known jailbreaks, detects: role-play attacks, hypothetical scenarios, encoded prompts. Block if confidence >0.85. Output: (1) Content filter: Re-check with moderation API, ensure response doesn\'t contain harmful content even if prompt was safe. (2) PII check: Scan response for user data, redact if leaked. (3) Fact-check: For factual claims (detected by LLM), query search API for verification. Flag if contradicted. (4) Hallucination detection: If response includes specific numbers/dates/names, validate against knowledge cutoff. Show uncertainty: "I believe X, but cannot verify." Balanced approach: (1) Tiered responses: Minor violations → warning + modified response. Major violations → refusal. (2) User override: "I understand the risks, proceed anyway" (logged for audit). (3) False positive handling: If user reports over-filtering, escalate to review, adjust thresholds. When check fails: (1) Input fail → "I can\'t help with that" + reason (if appropriate). (2) Output fail → Regenerate with stricter prompt, if fails again → generic safe response. (3) Log all failures for analysis. Post-hoc: Analyze blocked conversations weekly, identify patterns, improve filters.',
    keyPoints: [
      'Multi-layer: input moderation, output filtering, continuous monitoring',
      'Input: moderation API, PII detection, jailbreak classifier',
      'Output: re-check content, PII scan, fact-check claims, hallucination detection',
      'Balanced: tiered responses (warning vs refusal), user override option',
      'Handle failures: regenerate with stricter prompt, log for analysis',
    ],
  },
];
