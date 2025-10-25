export const ratiosMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'A company has gross margin of 60%, operating margin of 30%, and net margin of 15%. What does this margin structure reveal?',
    options: [
      'The company is highly profitable across all levels',
      'The company has strong pricing power (60% gross margin) but high operating expenses consume half the gross profit, and taxes/interest further reduce profitability to 15% net margin',
      'Operating expenses are too low relative to gross margin',
      'The company should focus on increasing gross margin',
      'All margins are equally important to analyze',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B. The margin waterfall shows: (1) Strong pricing power/low COGS (60% gross margin is excellent), (2) High OpEx consumes 30 points (60% → 30%), likely sales, marketing, R&D, (3) Interest/taxes consume another 15 points (30% → 15%). This is typical for high-growth tech companies: premium product (high gross margin) but heavy investment in growth (high OpEx). The analysis reveals where profit is being consumed at each stage.`,
  },

  {
    id: 2,
    question:
      'Company A has ROE of 20% with equity multiplier of 2.0. Company B has ROE of 20% with equity multiplier of 4.0. Which statement is most accurate?',
    options: [
      'Both companies are equally attractive since ROE is identical',
      'Company B is better because higher leverage boosts returns',
      'Company A has higher ROA (10%) than Company B (5%); Company A has superior operating performance, while Company B achieves same ROE only through higher leverage/risk',
      'Company B is more efficient at using assets',
      'The equity multiplier is irrelevant to ROE analysis',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. Using ROE = ROA × Equity Multiplier: Company A has ROA of 10% (20% / 2.0), Company B has ROA of 5% (20% / 4.0). Company A generates twice the operating returns per dollar of assets. Company B only matches ROE through 2x leverage. In a downturn, Company A is much safer - same returns, half the debt. Company B's higher leverage increases both returns AND risk.`,
  },

  {
    id: 3,
    question:
      'A SaaS company has magic number of 0.4 and LTV/CAC of 2.0x. What do these metrics indicate?',
    options: [
      'Excellent unit economics and efficient growth',
      'The company should invest more in sales and marketing',
      'Poor unit economics: Magic number <0.75 means S&M is inefficient (only $0.40 ARR per $1 spent), and LTV/CAC 2.0x is below 3.0x benchmark; company is likely burning cash unsustainably',
      'Both metrics are above industry average',
      'The company is ready for profitability',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. Magic number of 0.4 means every $1 in S&M spend only generates $0.40 in new ARR - very inefficient (good is >0.75). LTV/CAC of 2.0x is below the 3.0x benchmark, meaning lifetime value barely covers acquisition cost. Together, these signal: (1) Go-to-market strategy isn't working, (2) Customer acquisition is too expensive, (3) Company is likely unprofitable and burning cash, (4) Unit economics must improve before scaling. This company needs to fix efficiency before growing.`,
  },

  {
    id: 4,
    question:
      "A company's cash conversion cycle is 45 days. DSO is 30 days, DIO is 60 days. What is DPO, and what does this tell you?",
    options: [
      'DPO is 15 days; the company pays suppliers too quickly',
      'DPO is 135 days; calculation error',
      'DPO is 45 days (CCC = DSO + DIO - DPO → 45 = 30 + 60 - DPO → DPO = 45); company has moderate payment terms and reasonable working capital efficiency',
      'DPO cannot be determined from this information',
      'The cash conversion cycle formula is incorrect',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. Using CCC = DSO + DIO - DPO: 45 = 30 + 60 - DPO, therefore DPO = 45 days. This means: Company collects from customers in 30 days, holds inventory 60 days, pays suppliers in 45 days. Net effect: 45-day cash cycle. This is reasonable - the company pays suppliers 15 days after collecting from customers, providing a small working capital buffer. To improve, company could negotiate longer payment terms (increase DPO) or improve collections (decrease DSO).`,
  },

  {
    id: 5,
    question:
      'A bank has net interest margin (NIM) of 2.5%, efficiency ratio of 55%, and Tier 1 capital ratio of 12%. How should you interpret these metrics?',
    options: [
      'All three metrics are concerning and indicate problems',
      'The bank is poorly managed across all dimensions',
      'NIM 2.5% is reasonable (spread on loans), efficiency ratio 55% is moderate (means 55¢ of expenses per $1 revenue), Tier 1 capital 12% is strong (well above 8% minimum); overall solid bank metrics',
      'Only the capital ratio matters for bank analysis',
      "These metrics don't provide useful information",
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. For banks: (1) NIM of 2.5% means bank earns 2.5% spread between interest income and interest expense - typical for commercial banks, (2) Efficiency ratio of 55% means 55 cents in expenses per dollar of revenue - moderate (below 60% is good, below 50% is excellent), (3) Tier 1 capital ratio of 12% is strong (regulatory minimum is 8%, well-capitalized is 10%+) - shows bank can absorb losses. Overall: profitable operations, reasonable cost structure, strong capital cushion. This is a healthy bank.`,
  },
];
