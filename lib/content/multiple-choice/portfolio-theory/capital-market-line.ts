export const capitalMarketLineMC = {
    id: 'capital-market-line-mc',
    title: 'Capital Market Line - Multiple Choice',
    questions: [
        {
            id: 'cml-mc-1',
            type: 'multiple-choice' as const,
            question: 'The slope of the Capital Market Line represents:',
            options: [
                'The risk-free rate',
                'The market risk premium',
                'The Sharpe ratio of the market portfolio',
                'The beta of the market portfolio'
            ],
            correctAnswer: 2,
            explanation: 'Answer: C. CML slope = (R_M - R_f) / σ_M = Sharpe ratio of market portfolio. This represents the "price of risk" - the excess return earned per unit of risk. If slope = 0.44, each 1% of volatility commands 0.44% excess return. Not the risk-free rate (A), market risk premium is the numerator only (B), and beta = 1 for market by definition (D).'
        },
        {
            id: 'cml-mc-2',
            type: 'multiple-choice' as const,
            question: 'An investor wants to achieve 24% volatility using the Capital Market Line. The market portfolio has 18% volatility with 11% return, and the risk-free rate is 3%. What portfolio return should they expect?',
            options: [
                '12.7%',
                '13.7%',
                '14.7%',
                '15.7%'
            ],
            correctAnswer: 1,
            explanation: 'Using CML formula: E(R) = R_f + [(R_M - R_f)/σ_M] × σ_p = 3% + [(11%-3%)/18%] × 24% = 3% + [8%/18%] × 24% = 3% + 0.444 × 24% = 3% + 10.7% = 13.7%. Answer: B. This requires 133% allocation to market (leverage) since 24%/18% = 1.33. Borrow 33% at risk-free rate to invest.'
        },
        {
            id: 'cml-mc-3',
            type: 'multiple-choice' as const,
            question: 'According to two-fund separation theorem, all investors should hold:',
            options: [
                'Only the risk-free asset',
                'Only the market portfolio',
                'The same risky portfolio (market), differing only in mix with risk-free asset',
                'Different portfolios based on their views of expected returns'
            ],
            correctAnswer: 2,
            explanation: 'Answer: C. Two-fund separation states all investors hold the SAME optimal risky portfolio (the market/tangency portfolio), differing only in how much they allocate to it vs. risk-free asset. Risk-averse investors hold 70% market + 30% cash. Aggressive investors hold 130% market (30% leverage). The risky portfolio itself is identical for all investors.'
        },
        {
            id: 'cml-mc-4',
            type: 'multiple-choice' as const,
            question: 'A 1.5x levered market portfolio (150% market, -50% risk-free) will have:',
            options: [
                'Lower Sharpe ratio than unlevered market',
                'Same Sharpe ratio as unlevered market',
                'Higher Sharpe ratio than unlevered market',
                'Sharpe ratio depends on borrowing rate'
            ],
            correctAnswer: 3,
            explanation: 'Answer: D (realistically). Theoretically, if borrowing at risk-free rate, Sharpe stays constant (B). In practice, borrowing rates exceed risk-free rate (5% vs 3%), reducing Sharpe ratio. Example: Market Sharpe 0.44, but with 5% borrowing: (1.5×11% - 0.5×5% - 3%) / (1.5×18%) = (16.5% - 2.5% - 3%) / 27% = 11% / 27% = 0.41 < 0.44. Leverage costs reduce Sharpe.'
        },
        {
            id: 'cml-mc-5',
            type: 'multiple-choice' as const,
            question: 'The Capital Market Line applies to:',
            options: [
                'All portfolios and individual securities',
                'Only efficient portfolios that are combinations of the market portfolio and risk-free asset',
                'Only portfolios with beta = 1.0',
                'Any portfolio as long as it is well-diversified'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. CML applies ONLY to efficient portfolios combining market + risk-free asset. Individual stocks lie below CML due to idiosyncratic risk. Statement A is FALSE (individual securities not on CML). C is FALSE (CML portfolios can have any beta, depending on leverage). D is FALSE (even diversified portfolios may not be on CML unless they are optimal combinations).'
        }
    ]
};

