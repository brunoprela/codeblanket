import { MultipleChoiceQuestion } from '@/lib/types';

export const greeksDeltaGammaThetaVegaRhoMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'greeks-mc-1',
        question:
            'You own 10 ATM call contracts with delta of 0.50 each. To delta hedge, you should:',
        options: [
            'Buy 500 shares',
            'Sell 500 shares',
            'Buy 50 shares',
            'Sell 5,000 shares',
        ],
        correctAnswer: 1,
        explanation:
            'Delta hedging requires taking the OPPOSITE position in stock. Long calls have positive delta, so you need NEGATIVE delta (short stock) to neutralize. Calculation: 10 contracts × 100 shares per contract × 0.50 delta = 500 shares equivalent. To hedge: SELL 500 shares. This creates a delta-neutral position where stock moves don\'t affect your P&L (approximately). If stock goes up $1: Calls gain 10×100×0.50 = $500, Stock loses $500, Net ≈ $0. The hedge needs to be dynamic - as delta changes (gamma effect), you\'ll need to rehedge by buying/selling more shares.',
    },
    {
        id: 'greeks-mc-2',
        question:
            'Which statement about gamma is TRUE?',
        options: [
            'Gamma is highest for deep ITM options',
            'Gamma is always negative for put options',
            'Gamma is highest for ATM options',
            'Gamma decreases as time to expiration increases',
        ],
        correctAnswer: 2,
        explanation:
            'Gamma is HIGHEST for at-the-money (ATM) options. This is where delta changes most rapidly with stock price movements. Gamma is the same for calls and puts (both positive), not negative for puts. Deep ITM/OTM options have LOW gamma - their deltas are stable (near 1.0 or 0.0). Gamma INCREASES as expiration approaches (not decreases) - short-dated ATM options have explosive gamma. The gamma peak at ATM makes sense: there\'s maximum uncertainty about whether option expires ITM or OTM, so small stock moves dramatically change the probability (and therefore delta).',
    },
    {
        id: 'greeks-mc-3',
        question:
            'An option has theta of -$0.05 per day. Over a weekend (3 days), approximately how much value does it lose to time decay?',
        options: [
            '$0.05 (1 day)',
            '$0.10 (2 days)',
            '$0.15 (3 days)',
            '$0.00 (markets closed)',
        ],
        correctAnswer: 2,
        explanation:
            'Time decay occurs 24/7, even when markets are closed! Over a weekend (Friday close to Monday open), the option loses approximately 3 days of theta. Calculation: -$0.05 per day × 3 days = -$0.15 loss. This is why options often "gap down" on Monday morning - the weekend theta decay happens all at once when markets reopen. Many traders close short-dated positions before weekends to avoid this accelerated decay. Long option holders hate weekends (lose time value), while short option holders love them (collect 3 days of theta in one weekend). Theta decay is continuous in time, not dependent on market hours.',
    },
    {
        id: 'greeks-mc-4',
        question:
            'Implied volatility increases from 20% to 30%. An ATM call with vega of $8 will change in value by approximately:',
        options: [
            '+$0.80 (8% increase)',
            '+$8.00 (1% per vega)',
            '+$80.00 (10% increase)',
            '+$800.00 (10 points × $8)',
        ],
        correctAnswer: 2,
        explanation:
            'Vega measures price change per 1% change in volatility. Volatility changed from 20% to 30% = +10 percentage points. Option price change ≈ Vega × Vol Change = $8 × 10 = +$80. Note: This is per share, so for a full contract (100 shares), the position gains $80 × 100 = $8,000! This demonstrates why vega matters - volatility spikes (common during market crashes) create huge P&L swings. The VIX often moves 20-50% in a single day during stress, which would mean vega P&L of $160-$400 per contract for this option. Long options = long vega (profit from vol increases), short options = short vega (loss from vol increases).',
    },
    {
        id: 'greeks-mc-5',
        question:
            'Which Greek is LEAST important for short-term options traders?',
        options: [
            'Delta (direction)',
            'Gamma (curvature)',
            'Theta (time decay)',
            'Rho (interest rates)',
        ],
        correctAnswer: 3,
        explanation:
            'Rho (interest rate sensitivity) is the least important Greek for most traders, especially short-term. Why? Interest rates change slowly (Fed meets every 6 weeks, moves in 0.25% increments), and rho impact is small for short-dated options. Example: 1% rate increase might change an ATM 30-day option by $0.10-0.20, while a 1% stock move changes it by $0.50 (delta), or 1% vol increase changes it by $0.10 (vega). Rho DOES matter for: Long-dated options (LEAPS with 1+ years), Interest rate derivatives, Large institutional portfolios. But for typical options trading (30-90 days), focus on delta, gamma, theta, and vega. Rho is often ignored entirely by retail traders.',
    },
];

