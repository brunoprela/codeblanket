export const meanVarianceOptimizationMC = {
    id: 'mean-variance-optimization-mc',
    title: 'Mean-Variance Optimization - Multiple Choice',
    questions: [
        {
            id: 'mvo-mc-1',
            type: 'multiple-choice' as const,
            question: 'MVO is called an "error maximization machine" because:',
            options: [
                'It has many calculation errors in the optimization algorithm',
                'It amplifies small estimation errors in inputs into large weight errors',
                'It always produces sub-optimal portfolios',
                'It cannot handle more than 100 assets'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. MVO exploits estimation errors as if they were signal. Small 1% error in expected return can cause 20-30% weight changes. The optimizer treats noise as information and takes extreme positions based on it. Not about algorithm errors (A), it produces theoretically optimal portfolios for given inputs (C), and can handle 1000+ assets (D). The problem is: garbage in, garbage out - amplified.'
        },
        {
            id: 'mvo-mc-2',
            type: 'multiple-choice' as const,
            question: 'Which input to MVO has the highest estimation error?',
            options: [
                'Covariance matrix',
                'Expected returns',
                'Constraints',
                'All have equal error'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Expected returns have standard errors 2-3x larger than volatility estimates. With 5 years monthly data: return SE ≈ 2.6% annually, volatility SE ≈ 1.8%. Returns have low signal-to-noise ratio (~3-4), volatilities have higher (~10-11). This is why covariances are estimated more reliably than means, and why Black-Litterman/shrinkage focus on return estimates.'
        },
        {
            id: 'mvo-mc-3',
            type: 'multiple-choice' as const,
            question: 'Black-Litterman model improves upon MVO by:',
            options: [
                'Using faster optimization algorithms',
                'Starting with market equilibrium returns and blending with investor views',
                'Eliminating the need for covariance estimates',
                'Guaranteeing positive returns'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Black-Litterman starts with equilibrium returns (reverse-optimized from market weights), then incorporates investor views with appropriate uncertainty. This avoids starting from noisy historical means. Doesn\'t use faster algorithms (A), still needs covariance matrix (C), and definitely doesn\'t guarantee positive returns (D). The key innovation is using market equilibrium as prior.'
        },
        {
            id: 'mvo-mc-4',
            type: 'multiple-choice' as const,
            question: 'Shrinkage estimators for expected returns typically shrink toward:',
            options: [
                'Zero',
                'Risk-free rate',
                'Grand mean (average of all assets)',
                'Market return'
            ],
            correctAnswer: 2,
            explanation: 'Answer: C. Ledoit-Wolf and similar shrinkage pull extreme sample means toward the grand mean (average across all assets). Formula: μ_shrunk = δ×μ_grand + (1-δ)×μ_sample. Typical δ = 0.5-0.8. Not toward zero (A), risk-free rate (B), or specifically market return (D), though grand mean and market return are related. The intuition: extreme estimates are likely noise, pull them toward center.'
        },
        {
            id: 'mvo-mc-5',
            type: 'multiple-choice' as const,
            question: 'For a 200-stock portfolio, sample covariance matrix is "ill-conditioned" when:',
            options: [
                'Number of observations (T) is less than number of assets (N)',
                'Correlations are too high',
                'Volatilities differ too much',
                'Expected returns are negative'
            ],
            correctAnswer: 0,
            explanation: 'Answer: A. When T < N (more assets than observations), covariance matrix is singular/near-singular (non-invertible). Example: 200 stocks with 3 years monthly data = 36 observations. Matrix has N-T = 164 zero eigenvalues! This is the dimensionality curse. High correlations (B) and differing volatilities (C) cause numerical issues but not fundamental singularity. Negative returns (D) are irrelevant to matrix conditioning.'
        }
    ]
};

