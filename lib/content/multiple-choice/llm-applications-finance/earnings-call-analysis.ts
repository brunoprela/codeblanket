export const earningsCallAnalysisMultipleChoice = {
  title: 'Earnings Call Analysis - Multiple Choice',
  id: 'earnings-call-analysis-mc',
  sectionId: 'earnings-call-analysis',
  questions: [
    {
      id: 1,
      question:
        'What is the most significant advantage of analyzing the Q&A portion of earnings calls separately from prepared remarks?',
      options: [
        'Q&A sessions are longer and provide more data',
        'Q&A reveals authentic management responses under pressure and their handling of difficult questions, while prepared remarks are carefully scripted',
        'Q&A portions contain more financial metrics',
        'Q&A sessions are easier for LLMs to process',
      ],
      correctAnswer: 1,
      explanation:
        'The Q&A portion reveals authentic management behavior under pressure. Prepared remarks are carefully scripted and reviewed by legal/PR teams, but Q&A responses show how management handles challenging questions, which topics make them defensive or evasive, and where they show genuine confidence vs discomfort. LLMs can detect tone shifts, evasiveness (answering different questions than asked), and defensive language patterns that are highly informative. This unscripted insight often provides better signal than polished prepared remarks.',
    },
    {
      id: 2,
      question:
        'When comparing management guidance to actual results over multiple quarters to build a "credibility score," what pattern suggests the most concerning management behavior?',
      options: [
        'Consistently meeting guidance exactly on target',
        'Occasionally missing guidance due to external factors beyond control',
        'Consistently providing optimistic guidance that results in misses, suggesting poor judgment or intentional sandbagging',
        'Providing conservative guidance that results in beats',
      ],
      correctAnswer: 2,
      explanation:
        'Consistently optimistic guidance followed by misses suggests either poor operational visibility (management doesn\'t understand their business) or intentional over-promising (managing expectations poorly). Both are problematic. Conservative guidance with beats (option 3) is common "sandbagging" and actually indicates management discipline. Exact hits (option 0) might suggest gaming but isn\'t necessarily negative. Occasional misses from external factors (option 1) are expected and acceptable.',
    },
    {
      id: 3,
      question:
        'What is the primary risk of generating real-time trading signals from earnings calls as they occur (before the call concludes)?',
      options: [
        'Transcript quality issues and lack of full context can lead to misinterpretation and erroneous trading decisions',
        'Trading in real-time is prohibited by regulations',
        'LLMs cannot process audio fast enough',
        'Earnings calls are too short to generate meaningful signals',
      ],
      correctAnswer: 0,
      explanation:
        "Real-time transcription can have errors, and trading on partial information before hearing full context is risky. A seemingly negative comment might be clarified or contextualized later in the call. Additionally, LLMs might misread tone or miss sarcasm in live transcripts. While speed provides edge, acting on incomplete or misunderstood information can lead to costly errors. Real-time trading on earnings calls isn't prohibited (option 1), but the information quality and completeness risks are substantial.",
    },
    {
      id: 4,
      question:
        'When analyzing tone and sentiment in earnings calls, what linguistic pattern is typically the strongest negative signal?',
      options: [
        'Using technical financial terminology',
        'Management providing detailed explanations and transparency about challenges',
        "Evasive answers, deflection, or answering questions that weren't asked, especially around key concerns",
        'Management expressing optimism about future prospects',
      ],
      correctAnswer: 2,
      explanation:
        "Evasive answers and deflection are strong negative signals. When management avoids directly addressing analyst questions, especially on key topics like guidance, competition, or operational challenges, it suggests they're hiding problems or lack confidence. This is more concerning than technical language (option 0, which is normal) or optimism (option 3). Detailed transparency about challenges (option 1) is actually a positive signal showing management has clear understanding and plan.",
    },
    {
      id: 5,
      question:
        'What approach provides the most robust earnings call analysis for generating trading signals?',
      options: [
        'Analyzing sentiment of prepared remarks only',
        'Counting specific positive vs negative keywords',
        'Combining transcript sentiment analysis, management tone assessment, analyst question patterns, and comparing to guidance track record',
        'Only analyzing the final conclusion of the call',
      ],
      correctAnswer: 2,
      explanation:
        'Robust analysis requires multiple angles: transcript sentiment captures overall tone, management tone assessment identifies confidence or concern, analyst question patterns reveal what sophisticated investors worry about, and guidance track record provides historical context on management credibility. This multi-faceted approach catches signals that single-source analysis misses. Keyword counting (option 1) lacks nuance, prepared remarks only (option 0) miss critical Q&A insights, and conclusion only (option 3) misses the substance of discussion.',
    },
  ],
};
