export const foreignExchangeMultipleChoiceQuestions = [
    {
        id: 1,
        question: "EUR/USD is quoted at 1.0850. This means:",
        options: [
            "1 USD buys 1.0850 EUR",
            "1 EUR buys 1.0850 USD",
            "The euro is weaker than the dollar",
            "You need to sell 1.0850 EUR to buy 1 USD",
            "The exchange rate is 108.50%"
        ],
        correctAnswer: 1,
        explanation: "In EUR/USD, EUR is the BASE currency (left) and USD is the QUOTE currency (right). The quote tells you how many quote currency units per 1 base currency unit. So EUR/USD = 1.0850 means **1 EUR = 1.0850 USD**. If you want to buy 1,000 EUR, you need to pay 1,000 × 1.0850 = 1,085 USD. **Common mistake**: Thinking the first currency is always the one you're measuring in dollars. USD/JPY = 150 means 1 USD = 150 JPY (NOT 1 JPY = 150 USD). Always read as: 1 unit of BASE = X units of QUOTE.",
        difficulty: "beginner"
    },
    {
        id: 2,
        question: "You enter a carry trade: borrow JPY at 0.1% and invest in AUD at 4.5%. You hold for one year. AUD/JPY stays exactly flat (no FX movement). Your return is approximately:",
        options: [
            "0.1% (the borrowing cost)",
            "4.5% (the investment rate)",
            "4.4% (4.5% - 0.1%)",
            "45% (4.5% / 0.1%)",
            "0% because FX rate didn't move"
        ],
        correctAnswer: 2,
        explanation: "Carry trade return = **Investment Rate - Funding Rate** (when FX is unchanged). You earn 4.5% on AUD deposits but pay 0.1% on JPY borrowing = net 4.4% profit. **This is pure interest differential**. **Why it works**: You're essentially getting paid to hold a position. **Why it fails**: If AUD/JPY drops 10%, you lose -10% on FX + earn 4.4% carry = net -5.6% loss. **Real example**: 2008 crisis, JPY carry trades had earned ~4% annually for years, but lost 30%+ in weeks when JPY soared as a safe haven. Many hedge funds blew up.",
        difficulty: "intermediate"
    },
    {
        id: 3,
        question: "The forward EUR/USD rate (6 months) is 1.0750, while spot is 1.0850. This means:",
        options: [
            "EUR is expected to strengthen vs USD",
            "EUR is trading at a forward premium (market expects EUR to appreciate)",
            "EUR is trading at a forward discount (USD interest rates are higher than EUR rates)",
            "This is an arbitrage opportunity",
            "Forward rates are meaningless predictions"
        ],
        correctAnswer: 2,
        explanation: "Forward < Spot means **forward discount** for the base currency (EUR). This reflects **interest rate parity**: USD rates > EUR rates, so EUR trades cheaper in the forward market. **Why**: If you can earn 5% in USD vs 3.5% in EUR, arbitrageurs will borrow EUR (cheap), convert to USD, earn 5%, and lock in the reverse conversion at the forward rate. This arbitrage pushes forward EUR lower. **Formula**: Forward = Spot × (1 + r_quote) / (1 + r_base). If r_USD > r_EUR → Forward < Spot. **Not a prediction**: Forward rates reflect interest differentials, NOT expectations of future spot rates (though often confused).",
        difficulty: "advanced"
    },
    {
        id: 4,
        question: "On January 15, 2015, the Swiss National Bank (SNB) suddenly removed the EUR/CHF floor of 1.20. EUR/CHF immediately fell to 0.85 (-30%). For traders who were long EUR/CHF with 100:1 leverage, what happened?",
        options: [
            "They lost 30% of their investment",
            "Their positions were automatically stopped out at small loss",
            "They were wiped out completely and owed money to brokers (negative balances)",
            "The SNB compensated them for losses",
            "Nothing - currency pairs can't move that fast"
        ],
        correctAnswer: 2,
        explanation: "With **100:1 leverage**, a 30% adverse move = **3,000% loss on margin**. Example: $1,000 margin controlling $100,000 position. -30% = -$30,000 loss. Account goes to -$29,000 (NEGATIVE). **What happened**: 1) Brokers couldn't close positions fast enough (no liquidity), 2) Slippage was massive (spreads widened to 1,000+ pips), 3) Many retail traders ended up owing tens of thousands, 4) 2+ brokers went bankrupt unable to collect, 5) Legal battles for years. **Lesson**: Leverage amplifies losses, and in flash crashes, your stop loss is worthless (gaps through it). **FX leverage is dangerous** - you can lose more than your deposit.",
        difficulty: "advanced"
    },
    {
        id: 5,
        question: "The FX market trades $7.5 trillion daily, making it the largest financial market. Most of this volume comes from:",
        options: [
            "Retail traders using online platforms",
            "Companies hedging international trade (Boeing, Apple, etc.)",
            "Large banks and institutional players (interbank market)",
            "Central banks intervening to control exchange rates",
            "Tourists exchanging currency at airports"
        ],
        correctAnswer: 2,
        explanation: "**Interbank market** (banks trading with each other) accounts for ~40-50% of FX volume. Add institutional players (hedge funds, asset managers, corporations) and you get 90%+ of volume. **Breakdown**: Banks: ~$3-4T, Institutional: ~$2-3T, Retail: ~$300-500B (only ~5-7%). **Why banks dominate**: 1) Market making (providing liquidity), 2) Proprietary trading, 3) Client facilitation, 4) Arbitrage. **Retail is tiny**: Despite millions of traders, retail accounts for single-digit percentage. Most retail FX brokers actually trade AGAINST their clients (they're the counterparty, not routing to real market) - why? Because 80%+ of retail traders lose money.",
        difficulty: "intermediate"
    }
];

