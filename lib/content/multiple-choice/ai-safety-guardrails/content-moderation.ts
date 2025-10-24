/**
 * Multiple choice questions for Content Moderation section
 */

export const contentmoderationMultipleChoice = [
  {
    id: 'content-mod-mc-1',
    question:
      'Your content moderation system uses OpenAI Moderation API. A post is flagged with: {"hate": 0.35, "harassment": 0.55}. Default threshold is 0.50. What should you do?',
    options: [
      'Block the post since harassment score > 0.50',
      'Allow the post since hate score < 0.50',
      'Block the post since combined scores > 1.0',
      'Flag for human review since one score is borderline',
    ],
    correctAnswer: 0,
    explanation:
      "Harassment score (0.55) exceeds the threshold (0.50), so the post should be blocked. Each category is evaluated independently—you don't combine scores or average them. Option D (human review) could be valid for borderline cases, but 0.55 is above threshold.",
  },
  {
    id: 'content-mod-mc-2',
    question:
      'You implement semantic caching for content moderation. Two texts: "This is spam" and "This is spam!" have cosine similarity 0.98. What is the MAIN risk?',
    options: [
      'The exclamation point might change the intent',
      'Cache might return wrong result if policies changed',
      'Similarity calculation is computationally expensive',
      'Users might notice cached responses are faster',
    ],
    correctAnswer: 1,
    explanation:
      'The main risk with caching is policy changes—if you update moderation rules, cached results reflect old policies. You must invalidate cache when policies change. Option A is low risk (98% similarity is very high). Option C is opposite (caching SAVES computation). Option D is not a risk.',
  },
  {
    id: 'content-mod-mc-3',
    question:
      'Your system blocks 5% of posts as violations. Users can appeal. 30% of appeals are upheld (post was actually fine). What is the PRIMARY issue?',
    options: [
      'Users are abusing the appeal system',
      'False positive rate is too high (30% of blocks are wrong)',
      'Appeal process is too lenient',
      'System is not blocking enough violating content',
    ],
    correctAnswer: 1,
    explanation:
      'If 30% of appeals are upheld, that means 30% of blocks were false positives—the system is blocking too much legitimate content. This indicates thresholds are too strict. Option A is unlikely if appeals are reviewed fairly. Options C and D are backwards—the system is already too strict.',
  },
  {
    id: 'content-mod-mc-4',
    question:
      'You want to moderate non-English content. The toxicity model only supports English. What is the BEST approach?',
    options: [
      'Translate to English, then moderate',
      'Use a multilingual toxicity model',
      'Skip moderation for non-English content',
      'Use keyword matching in the original language',
    ],
    correctAnswer: 1,
    explanation:
      'Multilingual models (B) are best—they preserve cultural context and avoid translation errors. Option A loses context and introduces errors. Option C is unsafe. Option D (keywords only) misses nuanced toxicity.',
  },
  {
    id: 'content-mod-mc-5',
    question:
      'Your moderation system has 200ms latency. Users complain about slowness. What optimization gives the LARGEST latency reduction?',
    options: [
      'Use a faster programming language (Rust instead of Python)',
      'Implement semantic caching (80% cache hit rate)',
      'Run moderation checks in parallel instead of serially',
      'Reduce ML model size by 50% (faster inference)',
    ],
    correctAnswer: 1,
    explanation:
      'Semantic caching with 80% hit rate means 80% of requests complete in ~2ms (Redis lookup) vs 200ms. This is the largest reduction. Option A (language) might save 20-30%. Option C (parallel) saves maybe 30-40%. Option D (smaller model) saves maybe 40-50%. Caching wins by far.',
  },
];
