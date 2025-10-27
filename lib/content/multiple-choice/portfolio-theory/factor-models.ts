export const factorModelsMC = {
  id: 'factor-models-mc',
  title: 'Factor Models (Fama-French) - Multiple Choice',
  questions: [
    {
      id: 'fm-mc-1',
      type: 'multiple-choice' as const,
      question: 'Fama-French 3-factor model adds which factors to CAPM?',
      options: [
        'Momentum and Quality',
        'Size (SMB) and Value (HML)',
        'Profitability (RMW) and Investment (CMA)',
        'Volatility and Liquidity',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. FF3 = Market + SMB (Small Minus Big) + HML (High Minus Low book-to-market). SMB captures size effect, HML captures value effect. Not momentum/quality (A - those are other factors), not RMW/CMA (C - those are FF5 additions), not volatility/liquidity (D - not in standard FF model). FF3 explains ~85-90% of return variance vs CAPM's ~70%.",
    },
    {
      id: 'fm-mc-2',
      type: 'multiple-choice' as const,
      question:
        'A fund has beta_MKT = 1.1, beta_SMB = 0.6, beta_HML = -0.3. This suggests:',
      options: [
        'Large-cap value fund',
        'Small-cap value fund',
        'Large-cap growth fund',
        'Small-cap growth fund',
      ],
      correctAnswer: 3,
      explanation:
        'Answer: D. SMB=0.6 (positive) → small-cap tilt. HML=-0.3 (negative) → growth tilt (low book-to-market = high valuations). Combined: small-cap growth. Not large-cap (A/C - SMB positive means small-cap), not value (B - HML negative means growth not value). MKT beta 1.1 shows slightly aggressive market exposure.',
    },
    {
      id: 'fm-mc-3',
      type: 'multiple-choice' as const,
      question: 'In Fama-French regression, R² = 0.91 means:',
      options: [
        'The fund will return 91% over the period',
        '91% of return variance is explained by the factors',
        'The fund has 91% probability of outperforming',
        'Alpha is 0.91',
      ],
      correctAnswer: 1,
      explanation:
        'Answer: B. R² measures explanatory power: 91% of return variance is explained by the three factors (market, SMB, HML). Remaining 9% is alpha + idiosyncratic risk. High R² (>0.85) indicates most returns driven by factor exposures, not manager skill. NOT about absolute returns (A), probability (C), or alpha magnitude (D). R²=0.91 is high - portfolio is mostly systematic factor exposures.',
    },
    {
      id: 'fm-mc-4',
      type: 'multiple-choice' as const,
      question:
        'Historical factor premiums (1926-2023): Market ~8%, SMB ~3%, HML ~5%, Momentum ~8%. This suggests:',
      options: [
        'Momentum is riskier than value',
        'Small-cap and value tilts add ~8% combined',
        'Market factor dominates all others',
        'These premiums are guaranteed going forward',
      ],
      correctAnswer: 1,
      explanation:
        "Answer: B. SMB 3% + HML 5% = 8% combined premium (approximately additive). Not about momentum risk (A - can't infer from returns alone), market doesn't dominate (C - other factors matter), premiums NOT guaranteed (D - historical ≠ future). The 8% combined value+size premium explains why many successful funds tilt toward small-cap value.",
    },
    {
      id: 'fm-mc-5',
      type: 'multiple-choice' as const,
      question: 'Alpha after factor adjustment represents:',
      options: [
        'Total outperformance vs benchmark',
        'Market timing ability',
        'Return from factor exposures',
        'True skill after controlling for systematic factor exposures',
      ],
      correctAnswer: 3,
      explanation:
        "Answer: D. Alpha in factor model is intercept after controlling for all factor exposures. Represents genuine skill/selection beyond systematic factors. NOT total outperformance (A - that includes factor exposures), not solely market timing (B - broader than timing), not factor returns (C - that's the beta×factor terms). Positive significant alpha (t>2) is rare and valuable - indicates manager adds value beyond factor tilts.",
    },
  ],
};
