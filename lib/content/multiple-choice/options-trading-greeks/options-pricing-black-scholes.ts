import { MultipleChoiceQuestion } from '@/lib/types';

export const optionsPricingBlackScholesMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'options-pricing-black-scholes-mc-1',
        question:
            'In the Black-Scholes formula, what does N(d1) represent?',
        options: [
            'The probability the option expires in-the-money',
            'The hedge ratio (delta) of the option',
            'The present value of the strike price',
            'The volatility of the underlying stock',
        ],
        correctAnswer: 1,
        explanation:
            'N(d1) is the delta of the option - it represents the hedge ratio, or how much the option price changes per $1 change in the stock price. For a call option, delta = N(d1), which ranges from 0 to 1. This is also approximately the probability the option will be exercised (though N(d2) is the exact probability). N(d1) appears in the formula as the weight on the current stock price: C = S×N(d1) - K×e^(-rT)×N(d2). The first term S×N(d1) is the expected present value of the stock if the option is exercised. Delta is the most important Greek for hedging - market makers use it constantly.',
    },
    {
        id: 'options-pricing-black-scholes-mc-2',
        question:
            'A stock trades at $50. Using Black-Scholes, you calculate a 50-strike call (90 days, 5% rate, 25% vol) is worth $3.45. The market price is $4.20. What is the implied volatility approximately?',
        options: [
            'Lower than 25% (market price below model)',
            'Exactly 25% (matches model)',
            'Higher than 25% (market price above model)',
            'Cannot determine without more information',
        ],
        correctAnswer: 2,
        explanation:
            'The market price ($4.20) is ABOVE the Black-Scholes model price ($3.45). This means the market is pricing in MORE risk/volatility than the 25% you used. To make the BS model produce $4.20, you need to increase the volatility input above 25%. This higher volatility is the "implied volatility" - it\'s implied by the market price. The relationship: Higher volatility → Higher option value (more uncertainty = more value). Since market price > model price, implied vol > assumed vol. In practice, you\'d use Newton-Raphson to solve for the exact IV (probably around 30-32% in this case). This is why traders quote options in "vol" terms rather than dollar prices - it normalizes for different strikes/expirations.',
    },
    {
        id: 'options-pricing-black-scholes-mc-3',
        question:
            'Which Black-Scholes assumption is MOST violated in real markets?',
        options: [
            'No transaction costs',
            'Constant volatility',
            'European exercise only',
            'No dividends',
        ],
        correctAnswer: 1,
        explanation:
            'Constant volatility is the most severely violated assumption. In reality: (1) Volatility changes over time (volatility clustering - high vol follows high vol), (2) Volatility varies by strike (volatility smile/skew), (3) Volatility is correlated with stock price (leverage effect). Evidence: VIX index changes dramatically day-to-day, volatility smile is observed in every liquid options market, implied volatility ≠ realized volatility. While other assumptions are also violated (there ARE transaction costs, most US stock options are American not European, stocks DO pay dividends), these can be adjusted for relatively easily. Constant volatility is fundamental to BS and cannot be easily "fixed" without moving to stochastic volatility models like Heston or SABR. This is why traders focus heavily on implied volatility and volatility trading.',
    },
    {
        id: 'options-pricing-black-scholes-mc-4',
        question:
            'A stock paying a $2 dividend in 30 days trades at $100. For a 90-day call option, what stock price should you use in Black-Scholes?',
        options: [
            '$100 (current price)',
            '$98 (current - dividend)',
            '$98.00 minus PV of $2 dividend',
            '$100 adjusted by dividend yield',
        ],
        correctAnswer: 2,
        explanation:
            'For discrete dividends, subtract the present value of the dividend from the stock price. Calculation: PV(dividend) = $2 × e^(-r × t) where t = 30/365 years. With r=5%: PV = $2 × e^(-0.05 × 30/365) = $2 × 0.9959 ≈ $1.992. Adjusted stock price = $100 - $1.992 = $98.01. Use $98.01 in Black-Scholes, not $100. Why? On ex-dividend date, stock drops by dividend amount (approximately). Call options are worth less because stock will drop. By subtracting PV of dividend, we account for this expected drop. Option (b) "$98" is wrong because it doesn\'t discount the dividend to present value. Option (d) "dividend yield" is for continuous dividends, not discrete dividends. This adjustment is critical for accurate pricing of options on dividend-paying stocks.',
    },
    {
        id: 'options-pricing-black-scholes-mc-5',
        question:
            'Using 20% volatility, Black-Scholes prices a 100-strike call at $3.50. If volatility increases to 30%, what happens to the call price?',
        options: [
            'Decreases (inverse relationship)',
            'Stays the same (vol doesn\'t affect calls)',
            'Increases (direct relationship)',
            'Depends on whether call is ITM or OTM',
        ],
        correctAnswer: 2,
        explanation:
            'Call price INCREASES with volatility. This is always true for both calls and puts - higher volatility = higher option value. Why? Volatility represents uncertainty. More uncertainty means: Larger potential upside (stock could go much higher), Downside is still limited (can\'t lose more than premium). Asymmetric payoff benefits from volatility. The sensitivity is called "vega": positive vega means option gains value when volatility increases. For this example: 20% vol → $3.50, 30% vol → might be $5.00+ (50% increase). This relationship is monotonic - always increases with vol, regardless of moneyness. It\'s NOT "depends on ITM/OTM" - even deep OTM options benefit from higher vol (more chance of finishing ITM). This is why option sellers fear "vol spikes" - their short positions lose value rapidly when volatility increases (like during market crashes).',
    },
];

