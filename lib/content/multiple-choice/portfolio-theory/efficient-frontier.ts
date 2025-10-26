export const efficientFrontierMC = {
    id: 'efficient-frontier-mc',
    title: 'Efficient Frontier - Multiple Choice',
    questions: [
        {
            id: 'ef-mc-1',
            type: 'multiple-choice' as const,
            question: 'The efficient frontier is a curve in mean-variance space. What shape does it have?',
            options: [
                'Straight line',
                'Hyperbola',
                'Parabola',
                'Circular arc'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. The efficient frontier is a hyperbola in (σ, μ) space. This results from the quadratic relationship between portfolio weights, variance, and return. The hyperbola has a leftmost point (global minimum variance portfolio) and extends infinitely to the right. It is NOT a straight line (that would be the Capital Market Line with a risk-free asset), parabola, or circular arc.'
        },
        {
            id: 'ef-mc-2',
            type: 'multiple-choice' as const,
            question: 'The global minimum variance portfolio (GMVP) is special because:',
            options: [
                'It has the highest Sharpe ratio on the frontier',
                'It requires no expected return estimates, only covariance matrix',
                'It always has equal weights across all assets',
                'It lies at the intersection of efficient frontier and Capital Market Line'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. GMVP is unique because it only depends on the covariance matrix, not expected returns (which are noisy). Formula: w_GMVP = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1). This makes it more robust than other frontier portfolios. Statement A is FALSE (tangency portfolio has highest Sharpe). C is FALSE (weights are inverse to variance/covariance). D is FALSE (tangency portfolio is at CML intersection).'
        },
        {
            id: 'ef-mc-3',
            type: 'multiple-choice' as const,
            question: 'Corner portfolios in efficient frontier construction represent:',
            options: [
                'Portfolios with exactly two assets',
                'Points where constraints become active or inactive',
                'The minimum and maximum return portfolios',
                'Portfolios with equal weights'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Corner portfolios are points where the optimal solution structure changes - an asset enters/exits the portfolio or hits a constraint limit. Between corner portfolios, optimal weights change linearly. This allows efficient computation: calculate ~10-30 corner portfolios instead of 1000+ points. Statements A, C, and D are not what defines corner portfolios.'
        },
        {
            id: 'ef-mc-4',
            type: 'multiple-choice' as const,
            question: 'For a 100-asset portfolio, how many unique parameters must be estimated in the covariance matrix?',
            options: [
                '100',
                '5,050',
                '10,000',
                '1,000,000'
            ],
            correctAnswer: 1,
            explanation: 'Answer: B. Covariance matrix has N(N+1)/2 unique parameters: 100(101)/2 = 5,050 (N variances on diagonal plus N(N-1)/2 unique covariances). This is NOT 100 (A), not N² = 10,000 (C - that\'s total elements including duplicates), and definitely not 1,000,000 (D). This dimensionality problem is why factor models and shrinkage estimators are crucial - they reduce parameters to estimate.'
        },
        {
            id: 'ef-mc-5',
            type: 'multiple-choice' as const,
            question: 'Which computational complexity best describes solving mean-variance optimization for N assets?',
            options: [
                'O(N)',
                'O(N²)',
                'O(N³)',
                'O(2^N)'
            ],
            correctAnswer: 2,
            explanation: 'Answer: C. Quadratic programming for MVO typically requires O(N³) complexity due to matrix inversion/decomposition. For 1000 assets, this is 1 billion operations, taking seconds on modern CPUs. O(N) would be too fast (linear), O(N²) is matrix multiplication, and O(2^N) would be intractable (exponential). Modern solvers and Critical Line Algorithm can improve constants, but fundamental complexity remains cubic.'
        }
    ]
};

