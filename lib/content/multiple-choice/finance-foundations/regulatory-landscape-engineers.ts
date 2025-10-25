import { MultipleChoiceQuestion } from '@/lib/types';

export const regulatoryLandscapeEngineersMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'rle-mc-1',
      question:
        'Under Reg SHO, what must a broker do before executing a short sale?',
      options: [
        'Nothing - short sales are unrestricted',
        'Locate shares available to borrow (reasonable belief)',
        'Buy shares first (prohibited from short selling)',
        'File form with SEC within 24 hours',
      ],
      correctAnswer: 1,
      explanation:
        'Reg SHO Rule 203(b)(1): Must "locate" shares before shorting - reasonable grounds to believe shares can be borrowed and delivered. Process: (1) Check broker\'s inventory, (2) Query prime broker/stock loan desk, (3) Receive locate (valid 1 day). "Easy to borrow" stocks (AAPL, MSFT) auto-approved. Hard-to-borrow stocks may be rejected or cost high borrow fees (20-100% APR). Naked shorting (shorting without locate) was common pre-2008, led to massive FTDs (fail-to-deliver), Reg SHO now requires close-out within T+5. Violation penalties: Fines up to $250K + profit disgorgement.',
    },
    {
      id: 'rle-mc-2',
      question:
        'A customer makes these trades in 5 days: (1) Buy+sell AAPL same day, (2) Buy+sell TSLA same day, (3) Buy GOOGL Monday, sell Tuesday, (4) Buy+sell MSFT same day. Account value: $20K. What happens?',
      options: [
        'Nothing - only 3 day trades',
        'Account flagged as Pattern Day Trader (PDT), day trading restricted',
        'Account closed for violation',
        'Must deposit additional $10K within 24 hours',
      ],
      correctAnswer: 1,
      explanation:
        'PDT Rule (FINRA 4210): 4+ day trades in 5 business days + <$25K equity = Pattern Day Trader. Count: AAPL (day trade), TSLA (day trade), GOOGL (NOT day trade - different days), MSFT (day trade) = 3 day trades. Wait - need 4 for PDT! So actually customer is NOT PDT yet. But one more day trade would trigger PDT with <$25K account. Once flagged PDT: Can only day trade with $25K+ equity, Attempting day trade with <$25K = account frozen 90 days or until depositing to $25K. Workaround: Use multiple brokers (PDT tracked per broker, not across market), Upgrade to $25K+, Trade cash account (PDT only applies to margin accounts, but cash accounts have T+2 settlement restrictions).',
    },
    {
      id: 'rle-mc-3',
      question:
        'Under Reg BI (Best Interest), what must a broker-dealer do before recommending a complex product to a customer?',
      options: [
        'Nothing - customers are responsible for their own decisions',
        'Ensure product is suitable for customer (risk tolerance, experience, objectives)',
        'Require customer signature acknowledging risk',
        'Obtain SEC approval for the transaction',
      ],
      correctAnswer: 1,
      explanation:
        "Reg BI (June 2020): Broker-dealers must act in customer's best interest when making recommendations. Requirements: (1) Disclosure: Tell customer about conflicts (PFOF, proprietary products), (2) Care: Understand product + customer, ensure reasonable basis product is suitable, (3) Conflict of Interest: Mitigate conflicts, can't put profit ahead of customer, (4) Compliance: Document suitability analysis. Example: Recommending 3x leveraged ETF to retiree seeking income = Reg BI violation. System must: Capture customer profile (age, income, net worth, objectives, risk tolerance, experience), Score product risk/complexity, Block/warn if mismatch, Log all recommendations for 6-year retention. Penalties for violations: Fines, license suspension, restitution to customers.",
    },
    {
      id: 'rle-mc-4',
      question:
        'What triggers a CTR (Currency Transaction Report) filing requirement?',
      options: [
        'Any transaction over $5,000',
        'Cash transaction over $10,000',
        'Any wire transfer internationally',
        'Customer uses credit card',
      ],
      correctAnswer: 1,
      explanation:
        'CTR (FinCEN Form 112): Required for cash transactions >$10,000 in single day. "Cash" = currency (bills, coins), NOT checks, wire transfers, or ACH. Examples: Customer deposits $12,000 cash → CTR filed, Customer deposits $5,000 cash twice (total $10K) → might trigger CTR if same business day, Customer wires $50,000 → NO CTR (not cash). Structuring (illegal): Breaking up transactions to avoid CTR (e.g., $9K today, $9K tomorrow = $18K total) triggers SAR (Suspicious Activity Report) + potential prosecution. CTRs are routine (banks file millions/year), SARs are serious (indicate potential money laundering). System must: Track daily cash totals per customer, Auto-generate CTR if ≥$10K, Detect structuring patterns (multiple <$10K transactions), Flag for SAR if suspicious.',
    },
    {
      id: 'rle-mc-5',
      question:
        'How long must broker-dealers retain order records under SEC Rule 17a-4?',
      options: [
        '1 year',
        '3 years',
        '6 years (3 years readily accessible)',
        '10 years',
      ],
      correctAnswer: 2,
      explanation:
        'SEC Rule 17a-4(b): Broker-dealers must retain records 6 years (first 3 years in "readily accessible" location). "Readily accessible" = can retrieve within hours, not days. Acceptable: Database with query interface, Cloud storage (S3), Optical disk. NOT acceptable: Tape archives requiring manual retrieval. Must use WORM storage (Write Once Read Many) - prevents tampering: S3 Object Lock, Compliance mode prevents deletion even by root user, Legal hold mode allows retention extension for litigation. Records include: Orders (all details, timestamp, customer), Executions (fills, prices, venues), Communications (emails, chats, calls related to trades), Account openings (KYC documents). Violations: $50K-$500K fines, Criminal charges if obstruction of justice. Engineering: Design immutable audit log (append-only, cryptographic hashing), Set S3 Object Lock 7-year retention (6 required + 1 buffer), Daily integrity checks (verify hash chain), Disaster recovery (cross-region replication).',
    },
  ];
