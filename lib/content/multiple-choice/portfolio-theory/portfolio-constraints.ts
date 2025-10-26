export const portfolioConstraintsMC = {
    id: 'portfolio-constraints-mc',
    title: 'Portfolio Construction Constraints - Multiple Choice',
    questions: [
        {
            id: 'pc-mc-1',
            type: 'multiple-choice' as const,
            question: 'Which constraint reduces expected return most in typical portfolio optimization?',
            options: [
                'Turnover limits (≤50% annually)',
                'Position limits (no single stock >5%)',
                'Sector limits (match benchmark ±5%)',
                'Factor constraints (beta 0.9-1.1)'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Position limits prevent concentration in highest-conviction ideas. Unconstrained might put 30% in top stock; 5% limit forces diversification, reducing return ~2-4%. Turnover (A) costs ~0.5%, sector limits (C) ~0.3%, factor constraints (D) ~0.2%. Position limits most restrictive because alpha typically concentrated in top 10-20 stocks. Diversification improves risk but sacrifices return.'
        },
        {
            id: 'pc-mc-2',
            type: 'multiple-choice' as const,
            question: 'Shadow price (Lagrange multiplier) of 0.15 for sector constraint means:',
            options: [
                'The sector has 15% weight',
                'Relaxing constraint by 1% would increase return by 0.15%',
                'The constraint costs 15% of total return',
                'Sector has 15% risk contribution'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Shadow price is marginal value of relaxing constraint. λ=0.15 means increasing limit from 30% to 31% would boost expected return by 0.15% (15 bps). High shadow price indicates binding constraint significantly limiting performance. NOT the weight (A), not total cost (C), not risk contribution (D). Shadow prices guide where to relax constraints for maximum benefit.'
        },
        {
            id: 'pc-mc-3',
            type: 'multiple-choice' as const,
            question: 'ESG constraints requiring portfolio score ≥8/10 typically:',
            options: [
                'Have no impact on expected returns',
                'Increase tracking error by 2-5% vs unconstrained benchmark',
                'Reduce universe by 5-10% only',
                'Improve risk-adjusted returns'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. ESG ≥8 reduces universe by 50-75% (only top quartile), increases tracking error to 3-5% vs 1-2% for unconstrained. NOT no impact (A - reduces expected return 0.3-0.8%), not 5-10% reduction (C - much larger), evidence for improved returns (D) is mixed at best. ESG constraints involve meaningful trade-offs: values alignment costs performance (or at minimum, increases risk).'
        },
        {
            id: 'pc-mc-4',
            type: 'multiple-choice' as const,
            question: 'Transaction cost model with market impact should use:',
            options: [
                'Fixed cost per trade (e.g., 10 bps)',
                'Linear cost proportional to trade size',
                'Square root cost: cost ∝ √(trade size / daily volume)',
                'Cubic cost: cost ∝ (trade size)³'
            ],
            correctAnswer: 2,
            explanation: 'Answer: C. Market impact follows power law: cost ∝ (trade size / daily volume)^α where α ≈ 0.5-0.7. Square root model (Almgren-Chriss) is standard. NOT fixed (A - ignores trade size), not linear (B - too aggressive for large trades), not cubic (C - too extreme). Intuition: trading 10% of daily volume has more than 10x impact of 1% due to moving the market.'
        },
        {
            id: 'pc-mc-5',
            type: 'multiple-choice' as const,
            question: 'Regulatory constraint requiring "portfolio volatility ≤15%" is best implemented as:',
            options: [
                'Constraint in the optimization: w^T Σ w ≤ 0.0225',
                'Post-optimization check and rejection',
                'Penalty term in objective function',
                'Maximum position size of 5% per stock'
            ],
            correctAnswer: 0,
            explanation: 'Answer: A. Hard constraint in optimization ensures feasibility: w^T Σ w ≤ 0.15² = 0.0225. Optimizer finds best solution satisfying constraint. Post-check (B) wastes computation if solution infeasible. Penalty (C) allows violation (regulatory unacceptable). Position limits (D) may not guarantee 15% vol (depends on correlations). Quadratic constraint handled efficiently by QP solvers. Always encode regulatory constraints as hard constraints, not soft penalties.'
        }
    ]
};

