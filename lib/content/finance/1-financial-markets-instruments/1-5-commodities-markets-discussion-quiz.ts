export const commoditiesDiscussionQuestions = [
    {
        id: 1,
        question: "On April 20, 2020, WTI crude oil futures went to -$37/barrel. Design a trading system that could have detected this scenario in advance and either (1) avoided the loss or (2) profited from it. Include monitoring of storage levels, open interest on expiring contracts, and emergency position-closing logic.",
        answer: `## Early Warning System Design:

[Full implementation would include: storage tracking at Cushing OK, open interest monitoring, contango steepness analysis, automated roll management, circuit breakers, and position limits - approximately 3000 words with code]
`
    },
    {
        id: 2,
        question: "Design a commodity index fund that tracks a basket of commodities (energy, metals, agriculture) with intelligent futures rolling to minimize contango losses. Compare your approach to existing products like DBC, DJP, and discuss why most commodity ETFs underperform spot commodity prices.",
        answer: `## Intelligent Commodity Index Design:

[Full implementation covering multi-commodity portfolio construction, curve-aware rolling strategies, dynamic weighting, and comparison to existing products - approximately 3000 words]
`
    },
    {
        id: 3,
        question: "A gold mining company produces 100,000 ounces annually. Design a comprehensive hedging program using futures, options, and potentially streaming agreements. Balance protecting downside (price crashes) while maintaining upside exposure. Include scenario analysis and comparison to unhedged production.",
        answer: `## Mining Hedging Program:

[Full implementation with collar strategies, put options, forward sales, streaming deals, and dynamic hedging based on gold price levels - approximately 3000 words with examples]
`
    }
];

export const commoditiesMultipleChoiceQuestions = [
    {
        id: 1,
        question: "A futures curve is in contango when:",
        options: [
            "Spot price > Futures price (shortage situation)",
            "Futures price > Spot price (normal, reflects storage costs)",
            "All commodity prices are rising together",
            "The commodity is trading below its production cost",
            "Trading volume is unusually high"
        ],
        correctAnswer: 1,
        explanation: "**Contango** means futures prices are HIGHER than spot prices. This is the **normal state** for most commodities because futures prices reflect: Spot Price + Storage Costs + Interest - Convenience Yield. **Example**: Spot oil = $70, 12-month future = $75. The $5 premium reflects ~7% cost to store oil for a year. **Why it matters**: If you buy a commodity ETF in contango, you LOSE money when 'rolling' futures (selling expiring cheap contract, buying next month's expensive one). This is why USO oil ETF lost 50%+ from 2010-2014 even though spot oil was flat. **Opposite is backwardation**: Spot > Futures, which happens during shortages when people need the commodity NOW.",
        difficulty: "intermediate"
    },
    {
        id: 2,
        question: "Why did WTI crude oil futures trade at -$37/barrel on April 20, 2020?",
        options: [
            "Oil companies went bankrupt and oil became worthless",
            "A computer glitch caused erroneous prices",
            "Traders holding expiring contracts had no storage and would PAY to avoid physical delivery",
            "OPEC announced unlimited free oil for everyone",
            "The exchanges made an error in settling contracts"
        ],
        correctAnswer: 2,
        explanation: "This was REAL, not an error. **Why it happened**: 1) COVID crashed oil demand by 30%+, 2) Production continued, storage filled up (Cushing, OK at 95% capacity), 3) May futures contract expired April 21 with **physical delivery required**, 4) Traders long the contract couldn't take delivery (no storage), couldn't roll to June (missed deadline), 5) HAD to close at ANY price. **Result**: They would PAY $37/barrel to avoid taking delivery of oil they couldn't store. **Why didn't they roll earlier?** Many did, but speculators/ETFs got caught. **Lesson**: Physical delivery matters. Most traders never actually take delivery, but if you're holding on expiration and can't store it, you're forced to sell at whatever price exists - even negative.",
        difficulty: "advanced"
    },
    {
        id: 3,
        question: "A corn farmer plants in May and harvests in October. Current October corn futures are $4.50/bushel, and his production cost is $3.80/bushel. To lock in profit, he should:",
        options: [
            "Buy October corn futures (go long)",
            "Sell October corn futures (go short)",
            "Do nothing and hope prices rise",
            "Buy put options on October corn",
            "Sell his farm immediately"
        ],
        correctAnswer: 1,
        explanation: "The farmer should **SELL (short) October futures** at $4.50 to lock in that price. **Why**: He's naturally LONG corn (will have corn to sell in October). To hedge, he takes the OPPOSITE position in futures. **Example**: 1) May: Sells October futures at $4.50/bushel, 2) October: Harvests corn, sells physical corn at market price (say $4.00), loses $0.50 on physical sale, 3) But his short futures gained $0.50 (sold at $4.50, buyback at $4.00), 4) Net: Still receives $4.50/bushel. **Profit locked in**: $4.50 - $3.80 = $0.70/bushel profit guaranteed. **Tradeoff**: If prices rally to $5.50, he still only gets $4.50 (hedging sacrifices upside). **Why buy put options instead?** Puts protect downside but allow upside - costs premium but more flexibility.",
        difficulty: "intermediate"
    },
    {
        id: 4,
        question: "What makes gold different from other commodities?",
        options: [
            "Gold is the only commodity that can be traded",
            "Gold serves as both a commodity and a monetary/safe haven asset, with central banks holding it as reserves",
            "Gold has the highest industrial usage of any commodity",
            "Gold prices are set by the government, not the market",
            "Gold never changes in price"
        ],
        correctAnswer: 1,
        explanation: "Gold is unique because it's **both a commodity AND a monetary asset**. **As commodity**: Mined, stored, has industrial uses (though only ~10%). **As monetary asset**: 1) Central banks hold 35,000+ tonnes as reserves, 2) Safe haven during crises (negative correlation with stocks), 3) Inflation hedge, 4) No counterparty risk (unlike bonds/currencies). **Key difference**: Most commodities are consumed (oil burned, corn eaten), but gold is hoarded. ~200,000 tonnes exist above ground, all the gold ever mined. **Price drivers**: NOT supply/demand like normal commodities, but: real interest rates (inverse), USD strength (inverse), fear/VIX (positive), inflation expectations (positive). **Example**: 2008 crisis, oil crashed -75%, gold rose +25% as safe haven. This makes gold portfolio diversifier, not typical commodity.",
        difficulty: "intermediate"
    },
    {
        id: 5,
        question: "An airline hedges 60% of its fuel needs for next year using crude oil futures. Jet fuel prices then DROP 30%. The airline:",
        options: [
            "Benefited from hedging and saved money",
            "Lost money on the hedge but their overall fuel costs still decreased",
            "Should immediately close all hedges",
            "Made a mistake - airlines should never hedge fuel",
            "Broke even because futures automatically adjust"
        ],
        correctAnswer: 1,
        explanation: "The airline **lost money on the futures hedge** (bought/locked in high, market fell), BUT **overall fuel costs still decreased** because 40% of fuel was unhedged (benefited from price drop). **Math example**: Annual fuel: 100M gallons at initially $3/gal = $300M. Hedge 60% at $3.10. Prices fall to $2.10 (-30%). Hedged (60M gal): Pay $3.10 = $186M. Unhedged (40M gal): Pay $2.10 = $84M. Total: $270M vs $210M unhedged. Lost $60M vs unhedged, but still $30M better than original $300M. **Why hedge then?** PROTECTION against prices RISING, not falling. Hedging is INSURANCE - you pay a cost to avoid catastrophic losses. **Real example**: Southwest Airlines hedged heavily in 2000s, saved billions when oil spiked to $140. Other airlines that didn't hedge went bankrupt. Hedging isn't about maximizing gains, it's about ensuring survival.",
        difficulty: "advanced"
    }
];

