export const sharpeRatioPerformanceMC = {
    id: 'sharpe-ratio-performance-mc',
    title: 'Sharpe Ratio and Performance Metrics - Multiple Choice',
    questions: [
        {
            id: 'srp-mc-1',
            type: 'multiple-choice' as const,
            question: 'Fund A: 16% return, 22% vol. Fund B: 12% return, 15% vol. Risk-free rate 4%. Which has better risk-adjusted performance?',
            options: [
                'Fund A (Sharpe 0.55)',
                'Fund B (Sharpe 0.53)',
                'Equal performance',
                'Cannot determine without correlation'
            ],
            correctAnswer: 0,
            explanation: 'Fund A Sharpe = (16%-4%)/22% = 0.545. Fund B Sharpe = (12%-4%)/15% = 0.533. Answer: A. Fund A has marginally better risk-adjusted returns despite higher volatility. Correlation is irrelevant for Sharpe ratio calculation (only needed for portfolio combinations). Difference is small (0.545 vs 0.533), so practically similar performance.'
        },
        {
            id: 'srp-mc-2',
            type: 'multiple-choice' as const,
            question: 'M² (Modigliani-Modigliani) measure adjusts portfolios to:',
            options: [
                'Have the same expected return as benchmark',
                'Have the same volatility as benchmark',
                'Have the same Sharpe ratio as benchmark',
                'Eliminate systematic risk'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. M² levers/delevers portfolio to match benchmark volatility, then compares returns directly. Example: Fund with 20% vol, benchmark 15% vol → delever fund to 15% vol, then compare returns. Makes performance comparable on apple-to-apples risk basis. Not about matching returns (A), Sharpe (C), or eliminating systematic risk (D).'
        },
        {
            id: 'srp-mc-3',
            type: 'multiple-choice' as const,
            question: 'Treynor ratio differs from Sharpe ratio by using:',
            options: [
                'Downside deviation instead of total volatility',
                'Beta instead of total volatility',
                'Tracking error instead of total volatility',
                'Maximum drawdown instead of total volatility'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Treynor = (R_p - R_f) / β. Uses systematic risk (beta) instead of total risk (volatility). Appropriate for diversified investors where idiosyncratic risk is eliminated. Sortino uses downside deviation (A), IR uses tracking error (C), Calmar uses max drawdown (D). Treynor is the correct measure for portfolio components in well-diversified portfolios.'
        },
        {
            id: 'srp-mc-4',
            type: 'multiple-choice' as const,
            question: 'An active fund with 0.8 Information Ratio and 5% tracking error generates expected active return of:',
            options: [
                '2.0%',
                '4.0%',
                '6.25%',
                '10.0%'
            ],
            correctAnswer: 1,
            explanation: 'Active Return = Information Ratio × Tracking Error = 0.8 × 5% = 4.0%. Answer: B. An IR of 0.8 is excellent (>0.5 is considered very good). With 5% tracking error, this generates 4% expected outperformance. IRs above 0.5 are rare and difficult to sustain - most active managers struggle to achieve even 0.3-0.4.'
        },
        {
            id: 'srp-mc-5',
            type: 'multiple-choice' as const,
            question: 'Why is Sharpe ratio problematic for hedge funds with option-like payoffs?',
            options: [
                'It assumes returns are normally distributed',
                'It doesn\'t account for leverage',
                'It ignores fees',
                'It requires risk-free rate input'
            ],
            correctAnswer: 0,
            explanation: 'Answer: A. Sharpe ratio assumes returns are symmetric/normal. Hedge funds with options have skewed, fat-tailed distributions. Negative skew (frequent small gains, rare large losses) looks good on Sharpe but has hidden tail risk. Positive skew (frequent small losses, rare large gains) looks bad on Sharpe despite valuable properties. Use Omega or Sortino for asymmetric strategies.'
        }
    ]
};
