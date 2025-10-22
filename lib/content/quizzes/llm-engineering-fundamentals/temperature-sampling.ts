/**
 * Quiz questions for Temperature, Top-P & Sampling Parameters section
 */

export const temperaturesamplingQuiz = [
    {
        id: 'q1',
        question:
            'You are building a code generation feature and a creative writing feature in the same application. Explain how you would configure sampling parameters differently for each, and why these differences matter.',
        sampleAnswer:
            'For code generation, I would use temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, because: (1) Temperature=0 ensures deterministic output - code must be consistent and correct, not creative, (2) You want the highest probability tokens (most likely correct syntax), not diverse options, (3) Code often requires repetition (loop structures, repeated method calls) so penalties would harm quality, and (4) Testing is easier with deterministic output - same input always produces same code. For creative writing, I would use temperature=0.9, top_p=1.0, frequency_penalty=0.8, presence_penalty=0.6, because: (1) Temperature=0.9 enables creativity and varied word choice - essential for interesting prose, (2) Frequency_penalty=0.8 reduces word repetition - makes writing more engaging and professional, (3) Presence_penalty=0.6 encourages new topics and ideas - prevents circling back to same themes, and (4) Still want coherent output, so temperature not too high (keep under 1.2). Real impact: With temperature=0 for creative writing, you would get boring, repetitive text. With temperature=0.9 for code, you would get syntax errors and invalid constructs. The ~60% of cases where these parameters differ show why  task-specific configuration is critical - same model, different results based on parameters.',
        keyPoints: [
            'Temperature=0 for deterministic tasks (code, extraction)',
            'Temperature=0.7-1.0 for creative tasks (writing, brainstorming)',
            'Frequency penalty reduces repetition in creative content',
            'Code needs repetition (loops, patterns) so no penalties',
            'Match parameters to task requirements, not arbitrary values'
        ]
    },
    {
        id: 'q2',
        question:
            'Explain why OpenAI recommends altering either temperature OR top_p, but not both. What would happen if you set temperature=0.5 AND top_p=0.5?',
        sampleAnswer:
            'OpenAI recommends choosing one because both parameters control randomness in different ways, and combining them creates unpredictable interactions. How they work: Temperature reshapes the probability distribution - low temperature makes high-probability tokens even more likely (sharper distribution), high temperature flattens it (more uniform). Top_p filters tokens - keeps only top tokens whose cumulative probability = p, discarding the rest. The problem with both: If you set temperature=0.5 (sharpens distribution) AND top_p=0.5 (only keeps top 50% probability), you are applying two filters that interact non-linearly. For example, with temperature=0.5, the distribution becomes sharper - maybe the top token goes from 40% → 60% probability. Then top_p=0.5 might mean you are only sampling from 2-3 tokens instead of intended 10-15. This is unpredictable and varies by prompt. The compounding makes it hard to reason about outputs. Best practice: For controlled randomness, use temperature (easier to understand - 0 = deterministic, 1 = random). For filtering unlikely tokens, use top_p (useful when you want to prevent truly random tokens but keep distribution natural). Choose the one that matches your mental model. If you want "less random but not deterministic", use temperature=0.3-0.5. If you want "natural distribution but cut off long tail", use top_p=0.9. Do not combine unless you deeply understand both mechanisms and test extensively.',
        keyPoints: [
            'Temperature and top_p both control randomness',
            'Combining them creates non-linear, unpredictable interactions',
            'Temperature is more intuitive for most use cases',
            'Top_p filters unlikely tokens, temperature reshapes distribution',
            'Choose one based on your mental model and stick with it'
        ]
    },
    {
        id: 'q3',
        question:
            'A user complains that your chatbot sometimes gives identical responses to the same question, which feels robotic. However, code generation must be consistent. How would you solve this with sampling parameters?',
        sampleAnswer:
            'I would implement task-specific parameter configurations: For chatbot responses (where variety is desired), use temperature=0.7, presence_penalty=0.3, max_tokens=500. Temperature=0.7 provides natural variation - same question gets different phrasings, examples, or explanations while staying coherent. Presence_penalty=0.3 slightly encourages new topics/phrasings without forcing unnatural changes. Users get fresh responses that feel conversational. For code generation (where consistency is required), use temperature=0.0, no penalties, max_tokens=2000. Temperature=0 ensures identical input produces identical code - critical for reliability and testing. Code quality benefits from  determinism, not creativity. Implementation strategy: (1) Detect task type from user intent or explicit flags, (2) Maintain separate parameter configs per task type, (3) Load appropriate config before each request, and (4) For edge cases (code explanation), use temperature=0.3 - slight variation in explanation without changing code logic. Real example: User asks "How do I sort a list in Python?" With temperature=0: Always suggests [].sort() or sorted(). With temperature=0.7: Sometimes suggests sorted(), sometimes [].sort(), sometimes list comprehension with comparator, sometimes key functions. For explanation tasks, variety helps learning. For code tasks, consistency helps reliability. The key insight: Same application needs different parameters for different features - build this into architecture from day one.',
        keyPoints: [
            'Different features need different parameters',
            'Chatbot benefits from variation (temperature=0.7)',
            'Code generation requires consistency (temperature=0)',
            'Use task detection to route to appropriate configs',
            'Presence penalty adds variety without incoherence'
        ]
    }
];

