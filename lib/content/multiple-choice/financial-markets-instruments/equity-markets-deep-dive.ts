import { MultipleChoiceQuestion } from '@/lib/types';

export const equityMarketsDeepDiveMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fm-1-1-mc-1',
        question:
            'A stock has a bid of $50.00 with 500 shares and an ask of $50.10 with 800 shares. What is the spread in basis points (bps)?',
        options: ['10 bps', '20 bps', '50 bps', '100 bps'],
        correctAnswer: 1,
        explanation:
            'Mid-price = ($50.00 + $50.10) / 2 = $50.05. Spread = $50.10 - $50.00 = $0.10. Spread in bps = ($0.10 / $50.05) × 10,000 = 19.98 ≈ 20 bps. Basis points measure spread relative to price, not absolute dollars. A 10-cent spread is wide for a $50 stock (liquid stocks typically <5 bps).',
    },
    {
        id: 'fm-1-1-mc-2',
        question:
            'According to the Efficient Market Hypothesis (EMH), which form suggests that technical analysis (using past prices) cannot generate abnormal returns?',
        options: [
            'Strong form only',
            'Semi-strong form only',
            'Weak form only',
            'All three forms (weak, semi-strong, and strong)',
        ],
        correctAnswer: 3,
        explanation:
            'All three forms of EMH include the weak form, which states prices reflect all past price information. If weak-form holds, technical analysis cannot work. Weak form: past prices reflected. Semi-strong: all public info reflected. Strong: even private info reflected. Each form builds on the previous, so all three invalidate technical analysis.',
    },
    {
        id: 'fm-1-1-mc-3',
        question:
            'The S&P 500 announces Tesla will be added to the index (representing 1.5% of the index). Index funds tracking the S&P have $10 trillion AUM. Approximately how much forced buying will occur?',
        options: ['$15 billion', '$50 billion', '$150 billion', '$500 billion'],
        correctAnswer: 2,
        explanation:
            '$10 trillion × 1.5% = $150 billion of Tesla stock must be purchased by index funds at market close on the effective date. This forced buying creates significant price impact (historically 5-10% for large additions), which sophisticated traders front-run by buying at announcement and selling at inclusion.',
    },
    {
        id: 'fm-1-1-mc-4',
        question:
            'A market maker quotes AAPL with a 1-cent spread and trades $50M daily at 50% fill rate. If adverse selection costs $0.0001 per dollar traded, what is daily profit?',
        options: ['$100,000', '$250,000', '$400,000', '$500,000'],
        correctAnswer: 2,
        explanation:
            'Gross revenue: $0.01 spread × $50M volume × 50% fill = $250K. Adverse selection cost: $0.0001 × $50M = $5K. Net profit: $250K - $5K = $245K ≈ $250K (option B). However, realistic adverse selection is higher (~$0.0002-0.0005), making actual profit closer to $225-240K. Market makers profit from volume, earning fractions of pennies per share on millions of shares.',
    },
    {
        id: 'fm-1-1-mc-5',
        question:
            'Which market structure characteristic is most associated with high-frequency trading (HFT) dominance?',
        options: [
            'Continuous double auction with central limit order book',
            'Periodic call auctions (batch matching)',
            'Dealer market with market makers',
            'Dark pool with hidden orders',
        ],
        correctAnswer: 0,
        explanation:
            "Continuous double auction with CLOB enables HFT because: (1) Orders continuously interact in microseconds, (2) Speed advantage matters (race to best price), (3) Can cancel/replace rapidly. In contrast: Call auctions batch orders (speed doesn't matter), dealer markets have fixed quotes(less arbitrage), dark pools hide flow(harder to predict).HFT thrives on continuous, transparent, fast markets.",
    },
];
