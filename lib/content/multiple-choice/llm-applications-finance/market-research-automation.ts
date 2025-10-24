export const marketResearchAutomationMultipleChoice = {
  title: 'Market Research Automation - Multiple Choice',
  id: 'market-research-automation-mc',
  sectionId: 'market-research-automation',
  questions: [
    {
      id: 1,
      question:
        'What is the most effective way for LLMs to identify emerging competitive threats from non-traditional sources?',
      options: [
        'Only analyzing traditional industry competitor reports',
        'Monitoring job postings, patent filings, VC funding, and strategic narrative changes across multiple sectors to detect capability building and strategic intent',
        'Focusing exclusively on press releases',
        'Tracking only direct competitors in the same industry',
      ],
      correctAnswer: 1,
      explanation:
        'Emerging threats often come from adjacent or unrelated sectors (tech companies entering finance, DTC brands disrupting retail). Effective detection requires: analyzing job postings (revealing skill building), patent applications (technology direction), VC funding patterns, regulatory filings, and strategic narrative analysis. LLMs can connect dots between seemingly unrelated developments and detect capability + market access + intent. Traditional competitor focus (options 0, 3) misses disruption until too late.',
    },
    {
      id: 2,
      question:
        'When comparing LLM-generated industry forecasts to expert analyst forecasts, in what situations might LLMs actually produce superior forecasts?',
      options: [
        'LLMs are always superior to human experts',
        'LLMs are never as good as human experts',
        'LLMs excel at synthesizing vast data without cognitive biases, avoiding anchoring, and applying base rates that experts ignore due to compelling narratives',
        'LLMs are only useful for very short-term forecasts',
      ],
      correctAnswer: 2,
      explanation:
        'LLMs have advantages in specific situations: better at synthesizing vast information systematically, no anchoring to previous forecasts, can identify base rate patterns humans ignore, better probabilistic thinking than point estimates, and faster updating. Superior when processing large standardized datasets, identifying cross-industry patterns, and removing emotional biases. However, humans excel with deep domain knowledge, primary research, and paradigm shifts. Neither is always better (options 0, 1), and LLMs work for various timeframes (not just short-term, option 3).',
    },
    {
      id: 3,
      question:
        'What is the primary risk when using LLMs for due diligence to identify red flags in target companies?',
      options: [
        'LLMs cannot read financial documents',
        'Confirmation bias where the system finds problems everywhere because it was prompted to look for them, interpreting neutral information negatively',
        'LLMs are too slow for due diligence timelines',
        'Due diligence requires no technology assistance',
      ],
      correctAnswer: 1,
      explanation:
        'Primary risk is confirmation bias amplification. Instructing LLMs to "find red flags" creates incentive to interpret ambiguous information negatively, and accumulating concerns without context makes every company look risky. Mitigation: structure as hypothesis testing not flag hunting, require affirmative evidence of problems, compare to peer baseline, weight by materiality, use LLM for both positive and negative due diligence, and implement advocate/devil\'s advocate structure. LLMs can read documents (option 0) and are fast (option 2).',
    },
    {
      id: 4,
      question:
        'When automating competitive analysis, what approach provides the most actionable insights?',
      options: [
        'Simple feature comparison matrices',
        "Only analyzing direct competitors' financial statements",
        'Combining quantitative competitive metrics with qualitative LLM analysis of strategic positioning, narrative changes, and capability development over time',
        'Focusing solely on market share numbers',
      ],
      correctAnswer: 2,
      explanation:
        'Most actionable analysis combines: quantitative metrics (market share, growth rates, financial performance) with qualitative LLM analysis of strategic positioning (how companies describe themselves and strategies), narrative changes over time (shifts suggesting strategic direction), and capability building (skills, technologies, partnerships). This holistic view catches strategic shifts before they appear in numbers. Simple matrices (option 0), financial-only (option 1), or market-share-only (option 3) analysis miss strategic context.',
    },
    {
      id: 5,
      question:
        'What data source is most valuable for LLMs to identify early signs of industry disruption?',
      options: [
        'Historical financial data only',
        'Combination of developer community activity, startup funding patterns, job postings, patent filings, and strategic narrative analysis across sectors',
        'Traditional industry reports exclusively',
        'Stock price movements',
      ],
      correctAnswer: 1,
      explanation:
        "Early disruption signals come from diverse sources: developer community activity (emerging platforms), startup funding patterns (capital flowing to disruptors), job postings (capability building), patent filings (technology development), and cross-sector strategic narratives (companies expanding into new areas). LLMs excel at synthesizing these diverse signals to detect disruption before it's obvious. Historical financial data (option 0), traditional reports (option 2), and stock prices (option 3) are lagging indicators.",
    },
  ],
};
