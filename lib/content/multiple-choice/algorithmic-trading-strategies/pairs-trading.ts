export const pairsTradingMC = [
    {
        id: 'ats-5-mc-1',
        question:
            'Pair has correlation 0.85 but cointegration p-value 0.20. What does this mean?',
        options: [
            'Tradeable pair, 0.85 correlation is sufficient',
            'Not tradeable, spread is not stationary (p>0.05)',
            'Tradeable with tight stops',
            'Use Kalman filter to fix cointegration',
        ],
        correctAnswer: 1,
        explanation: `**Correct: Not tradeable, spread is not stationary (p>0.05).**

**Correlation vs Cointegration:**
- Correlation 0.85: Assets move together
- Cointegration p=0.20: Spread NOT stationary

**Why Not Tradeable:**
- Spread = price_A - β*price_B
- If p>0.05, spread doesn't mean-revert
- Will experience persistent divergences

**Bottom Line**: Require p<0.05 for cointegration.`,
    },
    {
        id: 'ats-5-mc-2',
        question:
            'Your pair has 22-day half-life. How long should you expect to hold positions?',
        options: [
            '11 days (half of half-life)',
            '22 days (one half-life)',
            '30-40 days (until full reversion)',
            '5 days (quick in/out)',
        ],
        correctAnswer: 2,
        explanation: `**Correct: 30-40 days (until full reversion).**

**Half-Life Math:**
- Half-life = time to revert halfway
- Spread at entry: z = ±2.0
- After 22 days: z = ±1.0 (half)
- After 44 days: z = ±0.5 (quarter)
- Exit target: z = ±0.5

**Holding Period = 1.5-2x half-life = 33-44 days**`,
    },
    {
        id: 'ats-5-mc-3',
        question:
            'Round-trip transaction costs are 10bps. Spread standard deviation is $2. What minimum z-score entry ensures profitability?',
        options: [
            'z = ±0.5',
            'z = ±1.0',
            'z = ±2.0',
            'z = ±3.0',
        ],
        correctAnswer: 2,
        explanation: `**Correct: z = ±2.0.**

**Calculation:**
- Enter at z = 2.0 → spread = 2σ = $4
- Exit at z = 0 → spread = 0
- Profit = $4 per unit
- Cost = 10bps = 0.10% × position
- On $100K position: cost = $100
- Profit = $4 × (shares) >> $100

**Z=±2.0 provides sufficient profit buffer.**`,
    },
    {
        id: 'ats-5-mc-4',
        question:
            'You have $1M capital, want 20 pairs. What is appropriate position size per pair?',
        options: [
            '$50K per pair (5% each)',
            '$100K per pair (10% each)',
            '$250K per pair (25% each)',
            '$500K per pair (50% each)',
        ],
        correctAnswer: 1,
        explanation: `**Correct: $100K per pair (10% each).**

**Portfolio Construction:**
- Capital: $1M
- Pairs: 20
- Allocation per pair: 10% = $100K

**Why 10%:**
- Diversification: 20 pairs × 10% = well-diversified
- Risk management: Single pair loss limited to 10%
- Capacity: Can run 20 pairs simultaneously

**Gross exposure: 20 × $100K = $2M (2.0x leverage)**`,
    },
    {
        id: 'ats-5-mc-5',
        question:
            'Your 20-pair portfolio has average pair correlation 0.60. What is your effective diversification?',
        options: [
            '20 independent bets',
            '10 independent bets',
            '5 independent bets',
            '1.6 independent bets',
        ],
        correctAnswer: 3,
        explanation: `**Correct: 1.6 independent bets.**

**Calculation:**
Effective N = N / (1 + (N-1) × ρ)
= 20 / (1 + 19 × 0.60)
= 20 / 12.4
= 1.6 bets

**Interpretation:**
- Think you have 20 pairs
- Actually have 1.6 independent bets
- High concentration risk!

**Solution: Diversify across sectors, target ρ<0.3.**`,
    },
];

