export const financialDocumentAnalysisMultipleChoice = {
  title: 'LLMs for Financial Document Analysis - Multiple Choice',
  id: 'financial-document-analysis-mc',
  sectionId: 'financial-document-analysis',
  questions: [
    {
      id: 1,
      question:
        'What is the primary advantage of using LLMs for MD&A (Management Discussion & Analysis) analysis compared to traditional keyword-based sentiment analysis?',
      options: [
        'LLMs process text faster than keyword methods',
        'LLMs can understand context, nuance, and identify sentiment shifts that keywords miss',
        'LLMs are cheaper to implement than keyword searches',
        'LLMs eliminate the need for financial domain knowledge',
      ],
      correctAnswer: 1,
      explanation:
        'The key advantage of LLMs is their ability to understand context and nuance. Traditional keyword-based methods might count "risk" mentions but miss whether management is dismissing risks ("minimal impact expected") or emphasizing them ("significant uncertainty"). LLMs can detect subtle sentiment shifts, understand conditional statements, and identify when tone changes between filings—capabilities that simple keyword matching cannot achieve.',
    },
    {
      id: 2,
      question:
        'When comparing 10-K filings across multiple years to detect "change signals," what type of change is typically most significant for investment decisions?',
      options: [
        'Changes in formatting and document structure',
        'Changes in the order of risk factors listed',
        'Changes in how management describes core business operations, strategy, or discusses persistent problems',
        'Changes in the length of the filing',
      ],
      correctAnswer: 2,
      explanation:
        'Changes in how management describes core business, strategy, or persistent problems are most significant because they reveal strategic shifts, emerging concerns, or deteriorating confidence. For example, if management moves from confidently discussing a business segment to using hedging language ("challenging environment," "working to address"), this signals problems before they fully appear in financials. Formatting, order, and length changes are typically less meaningful.',
    },
    {
      id: 3,
      question:
        'What is the main risk of using LLMs to extract specific financial metrics from earnings transcripts or filings?',
      options: [
        'LLMs process information too slowly for real-time trading',
        'LLMs may hallucinate numbers or misattribute metrics, requiring validation against structured data',
        'LLMs cannot read PDF documents',
        'LLMs are biased toward positive numbers',
      ],
      correctAnswer: 1,
      explanation:
        'LLMs can confidently generate plausible but incorrect numbers (hallucination) or misattribute correct numbers to wrong metrics (e.g., attributing EBITDA to revenue). This is dangerous in financial contexts where accuracy is critical. Best practice is to use LLMs for qualitative analysis and understanding while validating all numerical extractions against structured data sources (XBRL, data feeds) or using rule-based extraction for numbers. Never rely solely on LLM-extracted financial metrics for investment decisions.',
    },
    {
      id: 4,
      question:
        'In the context of analyzing annual letters to shareholders, what pattern would be most concerning when detected by an LLM across multiple years?',
      options: [
        'Increasing length of the letter each year',
        'Shifting from specific, quantified achievements to vague, aspirational language without concrete results',
        'Changing the letter format from narrative to bullet points',
        'Mentioning the same strategic priorities consistently',
      ],
      correctAnswer: 1,
      explanation:
        'Shifting from specific, quantified achievements to vague aspirational language is a red flag suggesting deteriorating performance or loss of strategic direction. Strong management typically provides concrete examples and metrics; when they retreat to generalities, it often indicates they lack positive specifics to share. Consistent strategic priorities (option 3) can be positive (focus and discipline), while format changes (option 2) are typically cosmetic.',
    },
    {
      id: 5,
      question:
        'When using LLMs to analyze Risk Factors sections in 10-K filings, what approach provides the most actionable investment insight?',
      options: [
        'Counting the total number of risk factors disclosed',
        'Identifying which risks are boilerplate (industry-standard) vs. company-specific, and tracking changes in how specific risks are described over time',
        'Ranking risks by the order in which they appear',
        'Measuring the total word count of the risk factors section',
      ],
      correctAnswer: 1,
      explanation:
        'The most valuable analysis distinguishes boilerplate risks (generic, unchanged, industry-standard) from company-specific risks, and tracks how specific risks evolve. For example, if "supply chain disruption" goes from a generic mention to a detailed multi-paragraph discussion with specific impacts, this signals a material emerging problem. New company-specific risks or expanding discussion of existing ones are leading indicators. Simply counting risks or words (options 0, 3) or relying on order (option 2) misses the substantive content.',
    },
  ],
};
