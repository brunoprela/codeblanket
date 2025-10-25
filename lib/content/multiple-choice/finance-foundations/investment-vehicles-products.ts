import { MultipleChoiceQuestion } from '@/lib/types';

export const investmentVehiclesProductsMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'ivp-mc-1',
        question:
            'An ETF is trading at $105 while its NAV (underlying holdings) is worth $103. What should an authorized participant do?',
        options: [
            'Buy ETF shares, redeem for underlying stocks (profit $2/share)',
            'Buy underlying stocks, create ETF shares, sell ETF (profit $2/share)',
            'Do nothing (ETF prices supposed to exceed NAV)',
            'Alert SEC (potential market manipulation)',
        ],
        correctAnswer: 1,
        explanation:
            'ETF at $105, NAV $103 = $2 premium. Arbitrage: (1) Buy underlying stocks for $103, (2) Deliver to ETF issuer in creation unit (50K shares), (3) Receive ETF shares, (4) Sell ETF shares at $105. Profit: ($105 - $103) × 50,000 = $100,000 (minus transaction costs ~$25K). This arbitrage keeps ETFs trading close to NAV. When ETFs trade at discount, APs do reverse: buy ETF, redeem for stocks, sell stocks. This mechanism is why ETF prices rarely deviate >0.5% from NAV for liquid ETFs like SPY. Illiquid ETFs can deviate more (1-5%) because arbitrage less profitable.',
    },
    {
        id: 'ivp-mc-2',
        question:
            'A mutual fund has $100M in stocks, $5M cash, $2M accrued expenses, and 10M shares outstanding. What is the NAV?',
        options: [
            '$10.00 per share',
            '$10.30 per share',
            '$10.50 per share',
            '$9.80 per share',
        ],
        correctAnswer: 1,
        explanation:
            'NAV = (Total Assets - Liabilities) / Shares Outstanding. Assets = $100M stocks + $5M cash = $105M. Liabilities = $2M expenses. NAV = ($105M - $2M) / 10M = $103M / 10M = $10.30. This NAV is calculated once daily after market close (4pm ET). Investors buying mutual fund shares get this NAV regardless of when during day they place order (all orders executed at 4pm NAV). This is different from ETFs/stocks which trade at real-time market prices. Mutual fund NAV moves only based on underlying holdings\' price changes, not supply/demand for fund shares.',
    },
    {
        id: 'ivp-mc-3',
        question:
            'What is the primary advantage of ETFs over mutual funds for taxable accounts?',
        options: [
            'Higher returns (better performance)',
            'Tax efficiency (avoid capital gains distributions)',
            'Lower risk (less volatile)',
            'Better diversification (more holdings)',
        ],
        correctAnswer: 1,
        explanation:
            'ETFs are MORE tax-efficient due to "in-kind redemptions." When mutual fund investor sells, fund must sell stocks → realizes capital gains → distributes to ALL shareholders (you pay taxes even if you didn\'t sell!). ETFs use in-kind redemptions: AP redeems ETF shares, receives actual stocks (not cash) → no sale → no capital gains. Result: ETFs rarely distribute capital gains. Example: Vanguard Total Stock Market - Mutual fund (VTSMX): 0.5% annual capital gains distribution. ETF version (VTI): 0% distribution most years. On $100K portfolio, this saves ~$200/year in taxes (0.5% × $100K × 40% tax rate). Over 30 years: $6K+ saved.',
    },
    {
        id: 'ivp-mc-4',
        question:
            'A target-date 2050 fund currently has 85% stocks, 15% bonds. In 10 years (2034), what allocation would you expect?',
        options: [
            'Same: 85% stocks, 15% bonds (allocation fixed)',
            'More stocks: 95% stocks, 5% bonds (growth phase)',
            'Less stocks: 70% stocks, 30% bonds (glide path)',
            'More bonds: 30% stocks, 70% bonds (near retirement)',
        ],
        correctAnswer: 2,
        explanation:
            'Target-date funds follow a "glide path" - gradually decreasing stocks, increasing bonds as target date approaches. 2050 fund in 2024 = 26 years to retirement. 2034 = 16 years to retirement. Typical glide path: 40+ years: 90% stocks. 20-30 years: 75-85% stocks. 10-20 years: 60-75% stocks. 0-10 years: 40-60% stocks. Retirement+: 30-40% stocks. So 2034 (16 years out): ~70% stocks, 30% bonds is reasonable. The fund automatically rebalances quarterly to maintain target glide path. By 2050, will be ~40-50% stocks. This auto-adjustment is the main value prop: investors don\'t need to manually rebalance as they age.',
    },
    {
        id: 'ivp-mc-5',
        question:
            'A robo-advisor charges 0.25% annual fee and invests in ETFs with 0.10% expense ratios. For a $100K portfolio, what is the total annual cost?',
        options: [
            '$100 (only ETF fees)',
            '$250 (only robo-advisor fee)',
            '$350 (both fees)',
            '$500 (fees + trading costs)',
        ],
        correctAnswer: 2,
        explanation:
            'Total cost = Robo-advisor fee + ETF fees. Robo fee: $100K × 0.25% = $250. ETF fees: $100K × 0.10% = $100. Total: $350 annually (0.35% total). Note: Trading costs (rebalancing, tax-loss harvesting) add another ~0.05-0.10%, so true all-in cost ~0.40-0.45% or $400-450/year. Compare to: Human advisor (1% + ETF 0.10% = 1.10% = $1,100/year), Active mutual fund (1.5% expense ratio = $1,500/year), DIY ETF portfolio (0.10% = $100/year but requires your time). At $100K, robo-advisors are good value. At $1M+, human advisor may justify cost for complex planning. At <$10K, DIY free platforms (Robinhood + free ETF trades) may be better.',
    },
];

