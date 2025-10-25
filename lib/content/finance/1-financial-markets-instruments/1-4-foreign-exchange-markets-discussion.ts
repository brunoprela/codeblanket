export const foreignExchangeDiscussionQuestions = [
    {
        id: 1,
        question: "Design a multi-currency payment platform (like Wise/TransferWise) that offers better FX rates than banks. Explain how you'd source FX rates, implement the 'matching' system to avoid actual FX trades, calculate your revenue model, handle regulatory requirements (money transmission licenses), and manage FX risk for unmatched positions.",
        answer: `## Comprehensive Platform Design:

### Business Model Overview

**Problem**: Banks charge 3-5% hidden markup on FX trades  
**Solution**: P2P matching + wholesale FX rates  
**Revenue**: 0.5-1% transparent fee  

### Architecture

[Full implementation details for multi-currency payment platform - approximately 3000 words with code examples, regulatory considerations, and risk management systems]

The platform would use peer-to-peer matching to minimize actual FX conversions, source wholesale rates from multiple providers, and maintain compliance across jurisdictions while managing residual FX risk through hedging strategies.
`
    },
    {
        id: 2,
        question: "A hedge fund trades the EUR/USD carry trade with 10:1 leverage. Design a risk management system that monitors: (1) position P&L, (2) margin requirements, (3) interest rate changes, (4) volatility spikes (VIX), and (5) triggers automatic position reduction. Include the 2015 SNB franc shock as a stress test scenario.",
        answer: `## Comprehensive Risk Management System:

[Full implementation for carry trade risk management system - approximately 3000 words covering real-time monitoring, margin calculations, volatility detection, and automatic de-risking protocols with code examples]

The system would implement multiple layers of protection including pre-trade risk checks, real-time P&L tracking, volatility-adjusted position sizing, and circuit breakers for extreme market events.
`
    },
    {
        id: 3,
        question: "Explain how you would build an FX price aggregator that collects quotes from 10+ liquidity providers, determines the best bid/ask, implements smart order routing, and handles latency/stale quote issues. Discuss the challenges of sub-millisecond quote aggregation and how to detect 'toxic flow' from latency arbitrageurs.",
        answer: `## Price Aggregation System Design:

[Full implementation for FX price aggregation platform - approximately 3000 words covering multi-provider integration, smart order routing, latency management, and toxic flow detection with production-grade code]

The aggregator would use WebSocket connections to multiple banks, implement latency-aware routing, and detect predatory behavior through statistical analysis of fill rates and quote updates.
`
    }
];

