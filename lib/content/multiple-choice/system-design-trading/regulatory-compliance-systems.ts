import { MultipleChoiceQuestion } from '@/lib/types';

export const regulatoryComplianceSystemsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'rcs-mc-1',
      question:
        'Why is a blockchain-style hash chain used in audit trail systems?',
      options: [
        'To improve query performance',
        'To reduce storage costs',
        'To detect any tampering or modification of historical records',
        'To comply with cryptocurrency regulations',
      ],
      correctAnswer: 2,
      explanation:
        "Hash chain detects tampering: each entry contains hash of previous entry. If anyone modifies historical record, hash won't match previous_hash in next entry → chain breaks → tampering detected. Verification: recalculate hashes, check chain integrity. Not for performance (actually slower), not for storage (adds overhead with hashes), not related to cryptocurrency (though same principle). Provides cryptographic proof that audit trail hasn't been modified since creation.",
    },
    {
      id: 'rcs-mc-2',
      question: 'What is wash trading and why is it illegal?',
      options: [
        'Trading with insufficient funds in account',
        'Buying and selling same security to create artificial volume',
        'Trading based on insider information',
        'Failing to report trades to regulators',
      ],
      correctAnswer: 1,
      explanation:
        "Wash trading: trader buys and sells same security (often within seconds/minutes) to create false impression of volume and liquidity. Illegal because: deceptive practice, misleads other market participants who see volume and think there's genuine interest, can manipulate closing prices. Detection: same user, same symbol, buy + sell within short window, similar quantities, repeated pattern. Not insufficient funds (that's margin violation), not insider trading (different violation), not failure to report (compliance violation).",
    },
    {
      id: 'rcs-mc-3',
      question:
        'What is the deadline for CAT (Consolidated Audit Trail) reporting in the US?',
      options: [
        'End of trading day',
        'Midnight EST same day',
        '8am EST next business day',
        'End of month',
      ],
      correctAnswer: 2,
      explanation:
        '8am EST next business day: CAT requires all US equity orders reported by 8am EST (T+1). Example: Orders on Monday must be reported by Tuesday 8am. Late penalties: $1,000+ per day. Best practice: generate reports at 6am (2-hour buffer) to handle any issues. Not same day (impossible for after-hours orders), not monthly (far too infrequent), not midnight (deadline is morning).',
    },
    {
      id: 'rcs-mc-4',
      question: 'Why must audit trail data be stored for 7-10 years?',
      options: [
        'To improve machine learning models',
        'Regulatory requirement for potential investigations',
        'To reduce storage costs through compression',
        'For competitive analysis of past trades',
      ],
      correctAnswer: 1,
      explanation:
        "Regulatory requirement: SEC requires 7 years minimum retention for audit trails. Regulators may investigate trades years after occurrence (fraud, manipulation, insider trading). Firm must produce complete audit trail on demand. FINRA Rule 4511: records must be preserved for 6 years (some 3 years), but best practice is 7-10 years. Not for ML models (historical trading patterns), not for cost reduction (storage costs money), not for competitive analysis (that's business intelligence, not compliance).",
    },
    {
      id: 'rcs-mc-5',
      question:
        'How does GDPR compliance affect regulatory reporting to US regulators (CAT)?',
      options: [
        'GDPR prohibits all reporting to US regulators',
        'Customer IDs must be pseudonymized (hashed) before reporting',
        'All EU data must be deleted immediately',
        'GDPR has no impact on trading compliance',
      ],
      correctAnswer: 1,
      explanation:
        'Pseudonymization required: GDPR restricts transferring EU resident data to US. Solution: hash customer IDs (one-way function) before CAT reporting. Maintains privacy (hash irreversible) while enabling linkage (same customer = same hash). Legal basis: Article 6(1)(c) - compliance with legal obligation (CAT reporting required by law). Not prohibit all reporting (that would make compliance impossible), not delete data (must retain for compliance), GDPR definitely impacts trading (data privacy vs regulatory reporting must be balanced).',
    },
  ];
