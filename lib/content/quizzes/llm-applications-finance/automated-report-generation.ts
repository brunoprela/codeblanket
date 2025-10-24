export const automatedReportGenerationQuiz = {
  title: 'Automated Report Generation Discussion',
  id: 'automated-report-generation-quiz',
  sectionId: 'automated-report-generation',
  questions: [
    {
      id: 1,
      question:
        'What is the appropriate balance between standardization and personalization in automated portfolio reports? How can LLMs be used to create reports that feel personal while maintaining compliance and consistency across thousands of clients?',
      expectedAnswer: `Should discuss: standardization ensures regulatory compliance, consistent methodology, and quality control across all reports, while personalization increases engagement and relevance for individual clients. LLMs enable "mass personalization"-applying consistent analytical framework while tailoring language, examples, and focus areas to individual circumstances. Implementation: maintain standard data analysis and metrics but customize narrative based on client profile (risk tolerance, age, goals, sophistication level), highlight holdings and performance most relevant to their objectives, address their specific questions or concerns, and adjust complexity of explanations to their knowledge level. Challenges: ensuring personalized content doesn't introduce compliance risks by making unsuitable recommendations, maintaining auditability when each report is unique, and preventing hallucinations that could include incorrect personal information. Best practice: template structure with variable personalization, human review of personalized sections for first use, AB test different personalization approaches to measure engagement without sacrificing quality.`,
    },
    {
      id: 2,
      question:
        'How should automated report generation systems handle periods of poor performance? Discuss the ethical considerations and regulatory requirements around tone, explanation, and recommendations when clients are losing money.',
      expectedAnswer: `Should cover: ethical imperative to be honest about losses while avoiding panic-inducing language, regulatory requirements mandate disclosing risks and past performance accurately, balance between explaining losses and appearing to make excuses, and importance of maintaining trust through transparency. LLM systems risk either understating problems (overly optimistic) or catastrophizing (alarmist tone). Proper approach: clearly state actual performance vs benchmark, provide factual explanation of market conditions and portfolio impacts, contextualize losses within historical volatility expectations, explain actions being taken or considered, and maintain consistent methodology in good and bad times. Avoid: blaming external factors without accountability, comparing only to worse performers, minimizing legitimate client concerns, or creating false urgency for changes. Regulatory: ensure loss disclosures meet requirements, maintain same level of detail and attention in down markets, document that reports accurately reflect reality not marketing spin. Human oversight: particularly important for large losses, unusual circumstances, or vulnerable clients. Ethical consideration: client's financial wellbeing may depend on understanding risks-clarity is respect.`,
    },
    {
      id: 3,
      question:
        'What are the risks and mitigation strategies for hallucination in automated financial reporting? How can systems verify that generated reports contain accurate numbers and appropriate recommendations?',
      expectedAnswer: `Should analyze: LLMs can confidently generate plausible but incorrect financial metrics, attribute wrong performance to wrong holdings, create fictitious market events to explain performance, or recommend actions inconsistent with client profile. Risks: regulatory violations if reports contain false information, liability for unsuitable recommendations, client losses from acting on incorrect data, and reputational damage from discovered errors. Mitigation strategies: structured data extraction where LLMs generate narrative around validated numbers not create numbers themselves, cross-reference all quantitative claims against source data, implement automated checks (returns must sum correctly, percentages must total 100%, referenced holdings must exist in portfolio), require second-pass verification where different model or rule-based system validates key facts, maintain human review for material facts and recommendations, version control to track what LLM generated vs what was verified, and clear documentation of data sources. Best practice: LLM generates natural language explanations of pre-calculated, verified data rather than calculating within generation process. Separate concerns: data computation (deterministic), analysis (LLM assisted), and narrative generation (LLM with validation). Never allow LLM to be sole source of truth for financial metrics that drive client decisions.`,
    },
  ],
};
