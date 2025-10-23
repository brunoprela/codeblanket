/**
 * Quiz questions for Chat Completions & Message Formats section
 */

export const chatcompletionsmessagesQuiz = [
  {
    id: 'q1',
    question:
      'Your chatbot is approaching the 16K token context limit after 20 turns of conversation. Compare and contrast three different truncation strategies (sliding window, summarization, and importance-based), and explain which you would choose for a customer support chatbot.',
    sampleAnswer:
      'For a customer support chatbot, I would choose importance-based truncation because it preserves critical context. Here is why each strategy matters: Sliding window (keep recent messages) is simple and fast but loses important earlier context like the customer problem description or account details mentioned at the start. Summarization (compress old messages) preserves more information but requires additional LLM calls for summarization, adding cost and latency, and risks losing specific details like error codes or product IDs that might be mentioned. Importance-based (keep messages with keywords, questions, or key information) is optimal for support because it retains: the initial problem description, any error messages or technical details, account/order information, previous solutions attempted, and user questions throughout the conversation. Implementation: Tag messages with keywords like "error", "order #", "problem", "issue", question marks, and sentiment markers. Keep system message, all tagged important messages, and last 5 turns. This ensures support agents (or the bot) always have critical context while staying under token limits. The trade-off is slightly more complex logic, but  it dramatically improves support quality compared to blindly truncating old messages.',
    keyPoints: [
      'Sliding window loses important early context',
      'Summarization adds cost and may lose details',
      'Importance-based preserves critical information',
      'Support chatbots need problem description and technical details',
      'Use keyword tagging to identify important messages',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain why using a detailed system prompt is critical for production LLM applications. Provide a bad example and a good example of a system prompt for a code review assistant.',
    sampleAnswer:
      'System prompts are critical because they set consistent behavior across all interactions without needing to repeat instructions in every user message, reducing token costs and ensuring uniform quality. Bad example: "You are helpful." This is too vague - the model does not know its specific role, what style to use, or what to focus on. It will give generic responses that vary in quality. Good example: "You are an expert code reviewer with 10 years of experience. Review code for: 1) Bugs and logic errors with specific examples, 2) Performance issues and suggest optimizations, 3) Security vulnerabilities following OWASP guidelines, 4) Code style and best practices per language standards, 5) Missing edge cases and error handling. Format: Provide specific line references, explain why each issue matters, suggest concrete fixes with code examples, prioritize issues as Critical/Important/Minor. Tone: Professional, constructive, educational. Always explain the reasoning behind your feedback." This system prompt works because it: Defines clear expertise level and role, Lists specific review criteria, Sets output format expectations, Defines tone and style, and Provides behavioral guidelines. The result is consistent, high-quality code reviews that users can rely on. The detailed system prompt is worth the extra ~100 tokens per conversation because it eliminates ambiguity and ensures every response meets quality standards.',
    keyPoints: [
      'System prompts ensure consistent behavior',
      'Vague prompts lead to inconsistent quality',
      'Specify role, tasks, format, and tone explicitly',
      'Detailed system prompts reduce need for user message instructions',
      'Extra tokens in system prompt save tokens and improve quality overall',
    ],
  },
  {
    id: 'q3',
    question:
      'You are building a chat application that needs to support multi-turn conversations. Describe the data structure and storage strategy you would use, including how to handle context window limits and conversation persistence.',
    sampleAnswer:
      'My conversation management system would include: Data structure: Each conversation has an ID, created timestamp, system prompt, messages array (role, content, timestamp, metadata like token count), total tokens counter, and metadata (user_id, conversation_type). Storage strategy: Store in database (PostgreSQL) with conversations table and messages table (one-to-many), index on user_id and created_at for fast retrieval, and cache recent conversations in Redis for instant access. Context management: Before each LLM call, calculate total tokens in conversation, if > 80% of limit (e.g., 12.8K of 16K), apply truncation strategy (keep system + recent N messages + tagged important messages), track which messages were truncated in metadata, and update token counter after each turn. Persistence: Save each message immediately after generation (do not lose partial conversations), implement auto-save every N turns, provide conversation history export, and allow conversation continuation after breaks. Code example: Store conversation state in class with methods for add_message, get_messages_for_llm (handles truncation), save_to_db, and load_from_db. This ensures conversations can grow indefinitely in storage while staying within LLM context limits, users never lose data, and application can scale to millions of conversations.',
    keyPoints: [
      'Separate storage (unlimited) from context window (limited)',
      'Use database for persistence and cache for performance',
      'Truncate intelligently before sending to LLM',
      'Track metadata like token counts and truncation status',
      'Implement immediate saving to prevent data loss',
    ],
  },
];
