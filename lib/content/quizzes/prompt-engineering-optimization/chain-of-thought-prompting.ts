/**
 * Quiz questions for Chain-of-Thought Prompting section
 */

export const chainofthoughtpromptingQuiz = [
    {
        id: 'peo-cot-q-1',
        question:
            'When should you use Chain-of-Thought prompting versus direct prompting? What types of tasks benefit most from CoT?',
        hint: 'Consider task complexity, reasoning requirements, accuracy needs, and token costs.',
        sampleAnswer:
            'Use CoT when: 1) Multi-step reasoning needed (math problems, logic puzzles, complex analysis); 2) Accuracy is critical and worth higher cost; 3) Debugging or analysis tasks requiring systematic thinking; 4) Planning and strategy problems; 5) Code generation where understanding context matters. DON\'T use CoT for: 1) Simple classification (sentiment, categories); 2) Direct extraction tasks; 3) Translation or summarization; 4) Tasks where speed/cost critical and accuracy less important. CoT adds 30-50% to token costs but improves accuracy on complex tasks by 30-50%. In production: Use CoT for high-value tasks where mistakes are costly, skip for high-volume simple tasks. Cursor uses CoT for code generation but not for simple completions. Test both approaches - sometimes direct prompting suffices even for seemingly complex tasks.',
        keyPoints: [
            'CoT for multi-step reasoning and complex analysis',
            'Skip CoT for simple classification/extraction',
            'CoT improves accuracy 30-50% on hard tasks',
            'Adds 30-50% token costs',
            'Use for high-value tasks where accuracy matters',
            'Test both approaches to validate benefit',
        ],
    },
    {
        id: 'peo-cot-q-2',
        question:
            'Explain the ReAct pattern (Reasoning + Acting). How does it differ from standard CoT and when would you use it?',
        hint: 'Think about tool use, iterative problem-solving, and multi-step workflows.',
        sampleAnswer:
            'ReAct interleaves Thought-Action-Observation cycles: Model thinks about what to do (Thought), executes action (Action), receives result (Observation), repeats until solved. DIFFERS FROM COT: CoT is pure reasoning ending in answer; ReAct involves actions (tool calls, API requests, function execution) between reasoning steps. WHEN TO USE: 1) Multi-tool workflows (search, calculate, database queries); 2) Iterative problem-solving requiring feedback; 3) Unknown information needing real-time lookup; 4) Complex tasks requiring trying different approaches. EXAMPLE: "Find Tokyo population and compare to NYC" requires Thought (need population data) → Action (search Tokyo) → Observation (37.4M) → Thought (need NYC data) → Action (search NYC) → Observation (8.3M) → Thought (compare) → Answer. Used by agents like Cursor for code edits, AutoGPT for autonomous tasks, research assistants for multi-source queries. Implementation requires function calling and state management.',
        keyPoints: [
            'Interleaves thinking and acting with tool use',
            'CoT is pure reasoning; ReAct involves actions',
            'Essential for multi-tool workflows',
            'Requires function calling capability',
            'Used in agentic systems like Cursor',
            'Enables iterative, feedback-driven solving',
        ],
    },
    {
        id: 'peo-cot-q-3',
        question:
            'What is self-consistency in CoT? How does generating multiple reasoning paths improve reliability?',
        hint: 'Think about majority voting, confidence estimation, and when diversity helps.',
        sampleAnswer:
            'Self-consistency generates multiple independent reasoning paths (typically 5-10) and uses majority vote for final answer. PROCESS: 1) Generate N solutions with higher temperature for diversity; 2) Extract final answer from each; 3) Take most common answer as consensus; 4) Confidence = frequency of consensus answer. BENEFITS: Catches errors - if one path makes mistake, others likely correct; increases reliability on problems with multiple valid reasoning approaches; provides confidence measure; reduces impact of random variations. WHEN HELPFUL: Critical accuracy needs; ambiguous problems with multiple solution paths; when single-path reasoning unreliable; budget allows extra API calls. TRADE-OFFS: Costs N times more tokens and time; diminishing returns after 5-10 paths. In production: Use for high-stakes decisions (medical, financial, legal assistance) but not routine tasks. Implement with parallel API calls for speed. Success rate improves from ~85% to ~95% on complex reasoning.',
        keyPoints: [
            'Generates multiple reasoning paths and votes',
            'Improves reliability by catching errors',
            'Provides confidence through consensus',
            'Costs N times more but much more reliable',
            'Use for high-stakes decisions',
            'Typically 5-10 paths optimal',
        ],
    },
];

