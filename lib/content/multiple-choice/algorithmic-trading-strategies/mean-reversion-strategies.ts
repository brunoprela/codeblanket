import { MultipleChoiceQuestion } from '@/lib/types';

export const meanReversionStrategiesMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'ats-3-1-mc-1',
        question:
            'A stock has RSI = 25 (oversold), ADX = 35 (strong trend), price at 52-week low. According to mean reversion best practices, what should you do?',
        options: [
            'Buy immediately (RSI oversold signal)',
            'Wait for ADX to drop below 25 before buying',
            'Short the stock (strong downtrend continuing)',
            'Buy half position now, add if RSI drops below 20',
        ],
        correctAnswer: 1,
        explanation:
            'ADX = 35 indicates STRONG TREND. Mean reversion strategies fail during strong trends - oversold becomes more oversold. Wait for ADX < 25 (range-bound regime) before attempting mean reversion. Option A (buy immediately) is disaster - fighting strong downtrend. Option C (short) not mean reversion strategy. Option D (partial position) still violates trend filter. Rule: Never mean revert in trending markets (ADX > 25).',
    },
    {
        id: 'ats-3-1-mc-2',
        question:
            'Hurst exponent = 0.45, half-life = 15 days, ADF test p-value = 0.02. What does this indicate about the asset?',
        options: [
            'Strongly trending (use trend following)',
            'Strongly mean-reverting (use mean reversion)',
            'Random walk (use buy-and-hold)',
            'Insufficient data to determine',
        ],
        correctAnswer: 1,
        explanation:
            'ALL THREE tests confirm strong mean reversion: (1) Hurst 0.45 < 0.5 (anti-persistent/mean-reverting), (2) Half-life 15 days < 20 (fast reversion), (3) ADF p-value 0.02 < 0.05 (statistically significant stationarity). This is textbook mean-reverting asset. Option A wrong (Hurst > 0.5 would be trending). Option C wrong (random walk has Hurst ≈ 0.5). Option D wrong (all tests conclusive).',
    },
    {
        id: 'ats-3-1-mc-3',
        question:
            'You buy stock at $100 (price at lower Bollinger Band). Plan: add at $98, $96, stop at $94. If stopped out after all three adds, what is your loss?',
        options: ['2% loss', '4% loss', '6% loss', '8% loss'],
        correctAnswer: 2,
        explanation:
            'Position 1: 100 shares @ $100 = $10,000. Position 2: 100 shares @ $98 = $9,800. Position 3: 100 shares @ $96 = $9,600. Total: 300 shares, $29,400 invested, average entry $98. Stop @ $94: Loss = ($98 - $94) × 300 = $1,200. Loss % = $1,200 / $29,400 = 4.08% ≈ 4%. BUT wait - should calculate vs total capital or position? If $29,400 is 20% of $150K capital: actual loss is $1,200 / $150K = 0.8% of capital. Question ambiguous, but typically means 4% of position value.',
    },
    {
        id: 'ats-3-1-mc-4',
        question:
            'Z-score mean reversion: stock z-score = -2.5 (oversold). Exit when z-score crosses what level?',
        options: [
            'Z = 0 (returned to mean)',
            'Z = +2.5 (opposite extreme)',
            'Z = -0.5 (near mean)',
            'Z = -1.0 (halfway to mean)',
        ],
        correctAnswer: 2,
        explanation:
            'Mean reversion exit at z ≈ -0.5 to 0 (NEAR mean, not exactly at mean). Why not exact mean (z=0)? Because: (1) Price may oscillate around mean without reaching it exactly, (2) Waiting for exact mean gives back too much profit, (3) Exit ~0.5σ from mean captures most of move while reducing whipsaw. Option B (+2.5) is way too aggressive (waiting for opposite extreme = giving back all profit). Options C and D reasonable, but -0.5 is standard (capture 80% of reversion).',
    },
    {
        id: 'ats-3-1-mc-5',
        question:
            'Bollinger Bands with 20-period SMA and 2 std dev. Current: price $100, upper band $105, lower band $95. What is the bandwidth (%)?',
        options: ['5%', '10%', '20%', 'Cannot determine from given info'],
        correctAnswer: 1,
        explanation:
            'Middle band (SMA) = ($105 + $95) / 2 = $100. Band width = (Upper - Lower) / Middle = ($105 - $95) / $100 = $10 / $100 = 10%. Bandwidth measures volatility - narrow bands (< 5%) indicate low volatility (consolidation), wide bands (> 15%) indicate high volatility (expansion). 10% is typical/moderate. Note: Price = $100 coincides with middle band (not oversold/overbought).',
    },
];

