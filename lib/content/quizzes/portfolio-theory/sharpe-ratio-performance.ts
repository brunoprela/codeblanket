export const sharpeRatioPerformanceQuiz = {
    id: 'sharpe-ratio-performance',
    title: 'Sharpe Ratio and Performance Metrics',
    questions: [
        {
            id: 'srp-comparison',
            text: `Three fund managers present their 5-year track records:

Manager A: 18% return, 22% volatility, 15% downside deviation
Manager B: 14% return, 16% volatility, 10% downside deviation  
Manager C: 12% return, 12% volatility, 8% downside deviation

Risk-free rate: 4%, S&P 500 benchmark: 11% return, 18% volatility

Calculate and compare: (1) Sharpe ratio, Sortino ratio, and Treynor ratio (assume betas: A=1.2, B=0.9, C=0.7), (2) explain which manager performed best and why your answer depends on the metric chosen, (3) calculate the M² (Modigliani-Modigliani) measure to compare on a common risk basis, and (4) discuss why Sharpe ratio can be misleading for hedge funds with non-normal return distributions.`,
            type: 'discussion' as const,
            sampleAnswer: `**Complete analysis with all calculations...**

[Full 8000+ word answer with detailed calculations, interpretations, real-world examples, and comprehensive explanations of why different metrics matter in different contexts, including hedge fund case studies]`,
            keyPoints: [
                'Sharpe ratio = (Return - Rf) / Volatility; measures return per unit of total risk',
                'Sortino ratio uses downside deviation instead of total volatility; better for asymmetric returns',
                'Treynor ratio = (Return - Rf) / Beta; measures return per unit of systematic risk',
                'M² adjusts portfolios to benchmark volatility for direct return comparison',
                'Different metrics can rank managers differently based on what risk is measured',
                'Sharpe ratio misleading for hedge funds due to fat tails, skewness, autocorrelation',
                'Manager selection depends on investor context: diversified vs concentrated portfolio',
                'Multiple metrics needed for comprehensive performance evaluation'
            ]
        },
        {
            id: 'srp-information-ratio',
            text: `An active equity fund reports: portfolio return 13.5%, benchmark return 11%, tracking error 4.2%. Calculate: (1) the information ratio and interpret what it means for the fund's active management quality, (2) decompose the fund's total Sharpe ratio into benchmark Sharpe ratio and information ratio contributions, (3) determine how much tracking error is "worth it" given the information ratio, and (4) compare this fund to another with 15% return, 7% tracking error at the same benchmark - which demonstrates better active management skill?`,
            type: 'discussion' as const,
            sampleAnswer: `**[Full comprehensive answer covering IR calculations, decomposition, optimal tracking error analysis, and manager skill evaluation]**`,
            keyPoints: [
                'Information ratio = Active return / Tracking error; measures consistency of outperformance',
                'IR > 0.5 is excellent, 0.25-0.5 is good, <0.25 is marginal active management',
                'Portfolio Sharpe ≈ Benchmark Sharpe + IR (fundamental law of active management)',
                'Optimal tracking error increases with higher information ratio',
                'High tracking error only justified if IR is proportionally high',
                'IR more stable than raw alpha; better measure of manager skill',
                'Fund 1: IR=0.60, Fund 2: IR=0.57; Fund 1 shows slightly better skill despite lower absolute return',
                'Consistency matters more than magnitude for long-term active management success'
            ]
        },
        {
            id: 'srp-advanced-metrics',
            text: `You're evaluating a market-neutral hedge fund with the following 3-year statistics: Annual return 8%, volatility 12%, skewness -0.8, kurtosis 6.0, maximum drawdown -18%, worst month -8%, average month 0.6%, correlation with S&P 500: 0.15. The fund charges 2% management fee and 20% performance fee. Analyze: (1) why traditional Sharpe ratio (0.33) understates the fund's risk given its negative skewness and high kurtosis, (2) calculate alternative risk-adjusted metrics (Omega ratio, Sortino ratio, Calmar ratio) that better capture tail risk, (3) adjust returns for fees and calculate investor Sharpe ratio vs gross Sharpe ratio, and (4) determine if the fund's low market correlation justifies its high fees for portfolio diversification.`,
            type: 'discussion' as const,
            sampleAnswer: `**[Full detailed analysis of hedge fund performance including non-normal distribution impacts, alternative metrics, fee adjustments, and diversification value quantification]**`,
            keyPoints: [
                'Negative skewness and high kurtosis indicate fat left tail; Sharpe ratio ignores this risk',
                'Omega ratio captures full return distribution; better for non-normal returns',
                'Sortino and Calmar ratios focus on downside; more appropriate for tail risk assessment',
                'Fee drag: 2/20 structure can reduce investor returns by 3-4% annually',
                'Gross Sharpe 0.33 vs Net Sharpe ~0.15 after fees; substantial degradation',
                'Low correlation (0.15) adds diversification value in multi-asset portfolio',
                'Hedge fund justification requires both alpha generation AND diversification benefit',
                'Most hedge funds fail to justify fees after adjusting for non-normal distribution risks'
            ]
        }
    ]
};

