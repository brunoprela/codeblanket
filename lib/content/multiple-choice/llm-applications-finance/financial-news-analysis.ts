export const financialNewsAnalysisMultipleChoice = {
  title: 'Financial News Analysis at Scale - Multiple Choice',
  id: 'financial-news-analysis-mc',
  sectionId: 'financial-news-analysis',
  questions: [
    {
      id: 1,
      question:
        'What is the primary challenge when processing the same news story reported by dozens of outlets simultaneously?',
      options: [
        'Storage limitations from duplicate articles',
        'Risk of over-reacting to a single piece of information amplified by repetition, treating it as multiple independent signals',
        'LLMs cannot process multiple articles at once',
        'Different outlets use different languages',
      ],
      correctAnswer: 1,
      explanation:
        'The main risk is mistaking repetition for confirmation. When 50 outlets report the same story, naive systems might treat this as 50 independent signals, creating false confidence through amplification rather than true confirmation. While widespread coverage indicates the story\'s importance and market reach, it doesn\'t provide 50x more information. Proper systems must deduplicate events, identify original sources, and separate "event significance" from "information novelty."',
    },
    {
      id: 2,
      question:
        'Why is detecting sarcasm and speculation in financial news particularly critical for LLM-based sentiment analysis?',
      options: [
        'Sarcasm is very common in financial news',
        'Undetected sarcasm can flip sentiment entirely, and speculation treated as fact generates false trading signals',
        'Financial regulations require sarcasm detection',
        'Sarcasm makes articles longer to process',
      ],
      correctAnswer: 1,
      explanation:
        'Sarcasm can completely flip sentiment (e.g., "Great job losing a billion dollars" is negative, not positive), while speculation treated as fact generates false signals. Financial articles often use phrases like "could suggest" or "may indicate" (speculation) vs stating facts. Headlines posed as questions ("Is Tesla Overvalued?") aren\'t statements. LLMs trained on general text may miss financial-specific nuance. Acting on misinterpreted sarcasm or speculation as fact causes costly trading errors.',
    },
    {
      id: 3,
      question:
        'When building a news aggregation system, what is the most effective way to handle source quality and credibility?',
      options: [
        'Treat all news sources equally to avoid bias',
        'Only use a single premium source',
        'Weight sources by track record, with Bloomberg/Reuters breaking news valued more than blog aggregators, while tracking unique information from each',
        'Automatically reject news from sources below a certain threshold',
      ],
      correctAnswer: 2,
      explanation:
        "Effective systems weight sources by credibility while still extracting unique information from all sources. Bloomberg breaking a story has more signal than blogs aggregating it, but even lower-tier sources sometimes have unique angles. The approach should identify original sources, weight primary sources heavily, use secondary coverage to gauge market attention/reach, and track each source's accuracy over time. Treating all equally (option 0) wastes credibility information, while single-source (option 1) misses coverage breadth.",
    },
    {
      id: 4,
      question:
        'What is the optimal strategy for balancing speed vs accuracy in news-based trading signals?',
      options: [
        'Always wait for maximum confirmation regardless of news type',
        'Always trade immediately on any news to capture first-mover advantage',
        'Use a tiered approach: immediate action for factual, high-impact news from tier-1 sources; staged response for complex news requiring interpretation',
        'Never use LLMs for time-sensitive trading decisions',
      ],
      correctAnswer: 2,
      explanation:
        'The optimal approach is tiered based on news characteristics. Unambiguous factual events (FDA approval, earnings beat) from reliable sources justify immediate action. Ambiguous or complex news (strategic shift announced) benefits from waiting for interpretation. Breaking news from unreliable sources requires confirmation. Implementation: highest-conviction signals get fast execution, medium signals get partial positions pending confirmation, low-confidence signals trigger research only. This balances first-mover advantage against false positive risk.',
    },
    {
      id: 5,
      question:
        "When analyzing financial news at scale, what metric best indicates a story's potential market impact?",
      options: [
        'The word count of the article',
        'Combination of source credibility, information novelty, coverage breadth, and relevance to known market sensitivities',
        'The number of companies mentioned',
        'The time of day the article was published',
      ],
      correctAnswer: 1,
      explanation:
        'Market impact assessment requires multiple factors: source credibility (tier-1 vs aggregator), information novelty (new fact vs rehash), coverage breadth (how widely reported, indicating market attention), and relevance to known sensitivities (Fed policy during rate concerns has high impact). Single metrics like word count (option 0), companies mentioned (option 2), or time (option 3) are insufficient. Systems should compute composite impact scores incorporating these dimensions, validated against historical price moves.',
    },
  ],
};
