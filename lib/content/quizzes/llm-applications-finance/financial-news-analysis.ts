export const financialNewsAnalysisQuiz = {
  title: 'Financial News Analysis at Scale Discussion',
  id: 'financial-news-analysis-quiz',
  sectionId: 'financial-news-analysis',
  questions: [
    {
      id: 1,
      question:
        'How should a news analysis system handle the challenge of the same story being reported by dozens of outlets within minutes? Discuss deduplication strategies, source quality weighting, and the risk of over-reacting to a single piece of information amplified by repetition.',
      expectedAnswer: `Should discuss: naive systems might treat 50 articles about same event as 50 independent signals, creating false confidence through repetition rather than confirmation. Deduplication requires identifying core event despite varied headlines and angles, using entity recognition and semantic similarity. Source quality matters-Bloomberg breaking story has more signal than blog aggregating it. Proper approach: identify original source, weight subsequent coverage by outlet credibility, extract unique information from each article not just sentiment, and recognize that widespread coverage indicates importance but not necessarily additional confirmation. Risk: ignoring repetition misses that market impact correlates with exposure/reach regardless of information novelty. Solution: separate "event significance" from "information novelty" in signal generation, track both consensus view (repeated) and contrarian takes (rare), and weight primary sources heavily while using secondary coverage to gauge market attention.`,
    },
    {
      id: 2,
      question:
        'What are the unique challenges LLMs face in detecting sarcasm, speculation, and conditional statements in financial news? Why are these nuances critical for accurate sentiment analysis and how can systems be designed to handle them?',
      expectedAnswer: `Should cover: financial news often uses cautious language ("could suggest," "may indicate") that LLMs might misinterpret as factual, sarcasm in opinion pieces or social media can flip sentiment entirely if not detected, conditional statements ("if the Fed raises rates") require understanding the condition may not occur, questions posed as headlines ("Is Tesla Overvalued?") aren't statements of fact, unnamed sources ("sources say") have different reliability than official statements, and forward-looking statements have different weight than current facts. Challenges: LLMs trained on general text may miss financial-specific nuance, context windows may miss earlier setup for sarcasm, and subtle skepticism in writing may read as neutral. Solutions: fine-tune on financial text with labeled sarcasm/speculation, implement claim verification to separate speculation from fact, extract and track conditionals separately, weight named sources higher than anonymous, and use sentiment confidence scores to flag ambiguous text for human review. Critical because acting on speculation as fact causes false signals.`,
    },
    {
      id: 3,
      question:
        'How should trading systems balance the speed advantage of early news detection against the accuracy advantage of waiting for confirmation and additional context? Discuss the optimal strategy for different types of news events.',
      expectedAnswer: `Should analyze: unambiguous factual events (FDA approval, earnings beat) justify immediate action as information is unlikely to change, ambiguous or complex news (strategic shift announced) benefits from waiting for analyst interpretation, breaking news from unreliable sources requires confirmation before action, and market-moving potential should inform urgency-5% expected move justifies more risk than 0.5% move. Strategy framework: immediate action for factual, verifiable, high-impact news from tier-1 sources; staged response for complex news (small initial position, add after confirmation); wait for official statement when dealing with rumors or leaks; use confidence thresholds where low-confidence signals trigger alerts not trades. Consider: early-mover advantage degrades quickly as news spreads, false positives from rushing are costly, market micro-structure means even "correct" signals can fail if liquidity is poor during initial reaction. Optimal: tier system where highest-conviction signals (rare) get fast execution, medium signals get partial positions pending confirmation, low-conviction signals trigger research/monitoring only. Different by asset class-liquid large-caps can handle fast execution, small-caps risk moving market with aggressive entry.`,
    },
  ],
};
