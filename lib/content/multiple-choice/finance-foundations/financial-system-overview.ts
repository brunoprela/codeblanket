import { MultipleChoiceQuestion } from '@/lib/types';

export const financialSystemOverviewMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'fso-mc-1',
        question:
            'What is the present value of $10,000 received in 5 years, assuming an 8% annual discount rate?',
        options: [
            '$6,806',
            '$8,000',
            '$10,000',
            '$14,693',
        ],
        correctAnswer: 0,
        explanation:
            'PV = FV / (1 + r)^n = $10,000 / (1.08)^5 = $10,000 / 1.469 = $6,806. This demonstrates time value of money: $10,000 in 5 years is worth only $6,806 today because you could invest $6,806 at 8% and have $10,000 in 5 years. The difference ($3,194) represents 5 years of opportunity cost at 8%.',
    },
    {
        id: 'fso-mc-2',
        question:
            'Two strategies: Strategy A returns 60% with 80% volatility. Strategy B returns 25% with 20% volatility. Assuming 4% risk-free rate, which has a better Sharpe ratio?',
        options: [
            'Strategy A (Sharpe = 0.70)',
            'Strategy B (Sharpe = 1.05)',
            'Both equal (Sharpe = 0.70)',
            'Cannot determine without more data',
        ],
        correctAnswer: 1,
        explanation:
            'Sharpe A = (0.60 - 0.04) / 0.80 = 0.56 / 0.80 = 0.70. Sharpe B = (0.25 - 0.04) / 0.20 = 0.21 / 0.20 = 1.05. Strategy B has better risk-adjusted returns (1.05 vs 0.70). Key insight: Strategy B is MORE EFFICIENT. You could leverage B to 3x (75% return, 60% volatility) and STILL have better Sharpe (1.18) than unleveraged A. This shows why "highest return" doesn\'t mean "best strategy".',
    },
    {
        id: 'fso-mc-3',
        question:
            'In the 2008 financial crisis, what happens to a bank with 30:1 leverage if its asset portfolio falls 5% in value?',
        options: [
            'Bank loses 5% of equity',
            'Bank loses 50% of equity',
            'Bank loses 150% of equity (bankrupt)',
            'Bank is unaffected (assets = liabilities)',
        ],
        correctAnswer: 2,
        explanation:
            'Leverage amplification: Bank has $30 in assets, $29 in debt, $1 in equity (30:1). If assets fall 5% → lose $1.50. Equity = $1.00 - $1.50 = -$0.50 (negative equity = bankrupt!). Loss as % of equity: -$1.50 / $1.00 = -150%. This shows why high leverage is dangerous: Small asset moves create massive equity swings. 30:1 leverage means a 3.3% asset drop = 100% equity loss. This was Lehman Brothers\' fate in 2008.',
    },
    {
        id: 'fso-mc-4',
        question:
            'Which financial market function is most important for efficient capital allocation?',
        options: [
            'Liquidity provision (easy buying/selling)',
            'Price discovery (aggregating information)',
            'Risk management (hedging)',
            'Transaction facilitation (payment systems)',
        ],
        correctAnswer: 1,
        explanation:
            'Price discovery is fundamental to capitalism. When markets aggregate information from millions of participants to determine fair prices, capital flows to its most productive uses. Example: Tesla stock rises from $50 to $250 → signal to market that EV transition is real → capital flows to Tesla (via stock issuance, cheaper debt) and other EV companies → society allocates resources to valuable technology. Without accurate price signals, capital is misallocated (see: Soviet Union\'s planned economy failures). Liquidity, risk management, and transactions are important but secondary to getting prices right.',
    },
    {
        id: 'fso-mc-5',
        question:
            'What is the main way fintech companies like Robinhood, Stripe, and Chime disrupted traditional finance?',
        options: [
            'Lower fees through better risk models',
            'Better user experience (UX) and technology',
            'More sophisticated financial products',
            'Higher interest rates on deposits',
        ],
        correctAnswer: 1,
        explanation:
            'Fintech won on TECHNOLOGY and UX, not finance expertise. Robinhood: Zero commissions + beautiful mobile app (vs clunky E*TRADE). Stripe: 7 lines of code to accept payments (vs months integrating with traditional processors). Chime: Open account in 2 minutes from phone (vs visiting branch with paperwork). Key insight for engineers: You don\'t need MBA to disrupt finance. You need: (1) Better UX (mobile-first, fast, intuitive), (2) APIs (programmatic access), (3) Lower cost structure (cloud, automation), (4) Data/ML (fraud detection, credit scoring). Traditional banks had financial expertise but terrible technology. Fintech had great technology and "good enough" finance knowledge. Technology won.',
    },
];

