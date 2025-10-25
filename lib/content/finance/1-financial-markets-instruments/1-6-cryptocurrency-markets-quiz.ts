export const cryptocurrencyMarketsQuiz = {
  title: 'Cryptocurrency Markets - Quiz',
  questions: [
    {
      id: 1,
      question:
        "Bitcoin has a maximum supply of 21 million coins, with approximately 19.5 million currently in circulation. If Bitcoin\'s price is $40,000, what is the difference between its market capitalization and fully diluted valuation (FDV)?",
      options: ['$20 billion', '$40 billion', '$60 billion', '$80 billion'],
      correctAnswer: 2,
      explanation:
        'Market Cap = Current Supply × Price = 19.5M × $40K = $780B. FDV = Max Supply × Price = 21M × $40K = $840B. Difference = $840B - $780B = $60B. This represents the value of Bitcoin yet to be mined (1.5 million coins × $40K).',
    },
    {
      id: 2,
      question:
        "A trader has a $100,000 account and wants to trade Bitcoin with a stop loss of 15% (typical for crypto's high volatility). If they're willing to risk 2% of their account on this trade, what should their position size be?",
      options: ['$6,667', '$10,000', '$13,333', '$15,000'],
      correctAnswer: 2,
      explanation:
        "Position Size = (Account Size × Risk %) / Stop Loss % = ($100,000 × 0.02) / 0.15 = $2,000 / 0.15 = $13,333. With a 15% stop, risking $2K requires a $13,333 position. This is only 13.3% of the account, reflecting crypto's high volatility requiring smaller positions.",
    },
    {
      id: 3,
      question:
        "In the 2022 crypto crisis, Three Arrows Capital\'s $10 billion failure triggered additional losses through leverage and interconnectedness. If 20 entities were each exposed to 3AC with 5x leverage, what was the approximate total system loss (initial + secondary losses)?",
      options: ['$20 billion', '$40 billion', '$60 billion', '$100 billion'],
      correctAnswer: 2,
      explanation:
        "Initial loss: $10B (3AC). Exposure per entity: $10B / 20 = $500M. With 5x leverage, each entity's amplified loss: $500M × 5 = $2.5B. Total secondary losses: $2.5B × 20 = $50B. Total system loss: $10B + $50B = $60B. This demonstrates how leverage amplifies contagion 6x!",
    },
    {
      id: 4,
      question:
        'A DeFi protocol offers 12% APR with daily compounding. What is the effective APY (Annual Percentage Yield)?',
      options: ['12.00%', '12.36%', '12.68%', '12.75%'],
      correctAnswer: 2,
      explanation:
        'APY = (1 + APR/n)^n - 1, where n = compounding frequency. APY = (1 + 0.12/365)^365 - 1 = 1.1268 - 1 = 0.1268 = 12.68%. Daily compounding adds 0.68 percentage points vs simple 12% APR. This is why DeFi protocols advertise APY (looks better!), while traditional finance often quotes APR.',
    },
    {
      id: 5,
      question:
        'Comparing centralized exchanges (CEX) like Coinbase to decentralized exchanges (DEX) like Uniswap, which statement is most accurate about their key trade-offs?',
      options: [
        'CEX has higher fees but better custody security than DEX',
        'DEX has faster execution but requires KYC unlike CEX',
        'CEX has better liquidity and speed but higher custody risk than DEX',
        'DEX has deeper liquidity but higher smart contract risk than CEX',
      ],
      correctAnswer: 2,
      explanation:
        'CEX advantages: Better liquidity (order books), faster execution (centralized matching), lower fees (0.1-0.5% vs 0.3%+ for DEX). CEX disadvantages: Custody risk (exchange holds your crypto - Mt. Gox, FTX), KYC required. DEX advantages: Self-custody (you control keys), permissionless. DEX disadvantages: Higher fees (including gas), slower (blockchain confirmation), variable liquidity. The fundamental trade-off is convenience/speed (CEX) vs custody/permissionless (DEX).',
    },
  ],
};
