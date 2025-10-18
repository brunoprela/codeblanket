/**
 * Quiz questions for Introduction to Design Problems section
 */

export const introductionQuiz = [
  {
    id: 'q1',
    question:
      'Why do design problems often require combining multiple data structures? Give an example.',
    sampleAnswer:
      "Design problems combine data structures because single structures rarely meet all requirements. For example, LRU Cache needs BOTH O(1) access by key AND O(1) update of access order. HashMap alone gives O(1) access but cannot track order efficiently. LinkedList alone tracks order but requires O(N) to find elements. Combined, HashMap stores key->node mappings for O(1) access, while doubly LinkedList maintains LRU order with O(1) move-to-front. Each structure compensates for the other's weakness. This pattern repeats: Min Stack needs stack operations + O(1) getMin (use extra stack), Rate Limiter needs fast lookup + time ordering (use HashMap + Deque). Real systems are complex and single data structures are too limited.",
    keyPoints: [
      'Single structures rarely meet all requirements',
      'LRU: HashMap for O(1) access + LinkedList for order',
      'Each structure compensates weakness of other',
      'Min Stack: main stack + tracking stack',
      'Pattern repeats across design problems',
    ],
  },
  {
    id: 'q2',
    question:
      'How do you decide between simplicity and optimal performance in design problems?',
    sampleAnswer:
      'I start by clarifying constraints: "What\'s the expected scale? Should I optimize for speed or maintainability?" If capacity is small (say 100 items), a simple list might suffice even if O(N) - it\'s readable and fast enough. But for production systems (10K+ items), I choose optimal structures even if more complex. For interviews, I state both: "Simple solution would be X with O(N), but optimal requires Y with O(1) - should I implement optimal?" This shows judgment. I also consider: Will code be maintained by others? (favor simplicity). Are there performance SLAs? (favor optimization). Can we profile first? (premature optimization warning). The answer is usually: start simple, optimize bottlenecks with data.',
    keyPoints: [
      'Clarify constraints and scale expectations',
      'Small scale: simplicity wins',
      'Large scale: optimal structures needed',
      'State both options in interviews',
      'Consider maintainability vs performance',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the most important questions to ask when given a design problem?',
    sampleAnswer:
      'Most important questions: (1) What operations are needed and their expected frequency? This determines what to optimize. (2) What are the performance requirements? "Should get() be O(1)?" (3) What\'s the expected scale? 100 items vs 1M items changes approach. (4) How should edge cases behave? Return null, throw exception, or default value? (5) Are there space constraints or is unlimited memory OK? (6) Should it be thread-safe? These questions prevent building wrong solution. For LRU Cache: "What\'s capacity limit? What does get() return for missing keys? When exactly is something \'recently used\'?" Good questions show engineering maturity and prevent rework.',
    keyPoints: [
      'What operations and their frequency?',
      'Performance requirements (time complexity)?',
      'Expected scale (100 vs 1M items)?',
      'Edge case behavior?',
      'Thread-safety requirements?',
    ],
  },
];
