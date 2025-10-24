export const earningsCallAnalysisQuiz = {
  title: 'Earnings Call Analysis Discussion',
  id: 'earnings-call-analysis-quiz',
  sectionId: 'earnings-call-analysis',
  questions: [
    {
      id: 1,
      question:
        'How can LLM analysis of earnings call Q&A sessions provide trading signals that traditional earnings analysis might miss? Discuss the information asymmetry between what management says in prepared remarks versus how they respond to challenging analyst questions.',
      expectedAnswer: `Should discuss: Q&A reveals management confidence or evasiveness through response patterns, defensive answers signal underlying problems not disclosed in prepared remarks, specific questions analysts ask indicate their concerns which may be prescient, management tone changes when under pressure reveal authenticity, non-answers to direct questions are highly informative red flags, repeated questions about same topic signal analyst skepticism that may be warranted, and qualitative cues like hesitation or over-explanation can be detected by LLMs analyzing transcript text. Trading edge comes from identifying divergence between prepared optimism and Q&A defensiveness before it appears in numbers.`,
    },
    {
      id: 2,
      question:
        'What are the challenges and ethical considerations of generating real-time trading signals from earnings calls as they occur? How should systems balance speed-to-market with accuracy and the risk of misinterpretation?',
      expectedAnswer: `Should cover: real-time transcription errors can lead to catastrophic misinterpretation and losses, LLMs may misread sarcasm or nuance in live transcripts, speed pressure incentivizes trading on partial information before full context emerges, market impact of automated systems all trading on same LLM signals creates self-fulfilling prophecies and volatility, front-running risk where early signal recipients profit at expense of later ones, ethical concerns about amplifying information advantage for sophisticated automated traders, regulatory implications of algorithmic trading on material information, need for human oversight before acting on machine-generated signals, and balancing first-mover advantage with prudent risk management. Mitigation: implement confidence thresholds, require signal confirmation from multiple analysis angles, use tiered response (alert vs auto-trade), maintain human review for large positions, and be transparent about use of AI trading systems.`,
    },
    {
      id: 3,
      question:
        'How can comparing management guidance accuracy over multiple quarters build a "management credibility score"? What are the implications of such scoring for investment decisions and what could go wrong with over-reliance on this metric?',
      expectedAnswer: `Should analyze: systematic tracking of guided vs actual results reveals management sandbagging or over-promising patterns, consistent beaters may be sandbagging (conservative guides) which is actually positive signal, consistent missers indicate either poor visibility or poor judgment, guidance accuracy varies by industry (volatile vs stable), external factors can cause misses despite good management, credibility scores should weight recent quarters more heavily, sector rotation can make previous guidance irrelevant, and comparing to peer guidance accuracy provides context. Risks: management may be honest but operating in volatile environment, punishing conservatism incentivizes aggressive guidance which is worse, over-optimization on guidance accuracy ignores business fundamentals, mechanical rules miss context like industry disruption, and reduces complex management evaluation to single metric. Use guidance tracking as one input among many, not deterministic signal.`,
    },
  ],
};
