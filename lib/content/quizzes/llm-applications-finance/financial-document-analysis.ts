export const financialDocumentAnalysisQuiz = {
  title: 'LLMs for Financial Document Analysis Discussion',
  id: 'financial-document-analysis-quiz',
  sectionId: 'financial-document-analysis',
  questions: [
    {
      id: 1,
      question:
        'How can large language models fundamentally change the traditional financial analyst workflow for analyzing SEC filings? Discuss the transformation in terms of speed, comprehensiveness, and potential limitations. What human skills remain irreplaceable despite LLM capabilities?',
      expectedAnswer: `Should discuss: automation of document parsing and initial analysis, ability to process thousands of filings simultaneously vs human capacity for dozens, LLMs identifying patterns across company filings over time, extraction of structured data from unstructured text, rapid comparative analysis. Limitations: need for human judgment on materiality, understanding business context that may not be explicit in filings, verification of LLM extractions especially for critical numbers, identifying subtle red flags that require industry expertise, and maintaining skepticism about management narratives. Human irreplaceable skills: strategic thinking, understanding broader economic context, relationship building for primary research, and ethical judgment.`,
    },
    {
      id: 2,
      question:
        'What are the critical challenges and potential failure modes when using LLMs to extract financial metrics from SEC filings? How can these risks be mitigated while maintaining the efficiency benefits of automation?',
      expectedAnswer: `Should cover: hallucination of financial metrics (LLMs generating plausible but incorrect numbers), confusion between historical periods when multiple years are discussed, misinterpretation of pro forma vs GAAP numbers, difficulty with complex table structures in HTML/PDF, context loss in long documents leading to incorrect associations, ambiguity in identifying which entity metrics belong to (parent vs subsidiary). Mitigation strategies: cross-referencing LLM extractions with actual XBRL data, implementing validation rules for metric reasonableness, using multiple passes or models for critical metrics, maintaining human review for material numbers, employing structured extraction with explicit section identification, and creating detailed audit trails of how metrics were extracted for verification.`,
    },
    {
      id: 3,
      question:
        'Compare the map-reduce pattern for processing long financial documents versus feeding entire documents into large context window models (100k+ tokens). What are the trade-offs, and when would you choose each approach?',
      expectedAnswer: `Should analyze: Map-reduce advantages include explicit control over chunking strategy, ability to process documents exceeding any context window, parallel processing of chunks for speed, and clearer attribution of which section generated which insight. Disadvantages include potential loss of cross-document relationships, overhead of combining analyses, and difficulty maintaining coherent narrative across chunks. Large context window advantages: maintains full document context, better at understanding relationships across sections, simpler implementation, and more coherent final analysis. Disadvantages: very high token costs for long documents, slower processing, potential attention degradation at extreme lengths, and less explicit control over which sections influenced conclusions. Choose map-reduce for very long documents (300+ pages), when cost is a concern, when parallel processing is needed, or when clear section attribution is required. Choose full-context when document length permits (<100 pages), when cross-section insights are critical, when processing time is less important, and when willing to pay for premium performance.`,
    },
  ],
};
