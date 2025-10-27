export const completeTradingSystemQuiz = [
  {
    id: 'complete-trading-system-q-1',
    question:
      'Design the complete architecture for a trading system that handles 10,000 orders/day with <100ms latency. What are the key components and how do they interact?',
    sampleAnswer:
      'Trading System Architecture:\n\n' +
      '**Component Stack:**\n\n' +
      '1. **Order Management System (OMS)**:\n' +
      '   - PostgreSQL for order persistence\n' +
      '   - REST API for traders\n' +
      '   - Order validation and routing\n' +
      '   - Latency budget: 20ms\n\n' +
      '2. **Execution Management System (EMS)**:\n' +
      '   - FIX engine (QuickFIX) for broker connectivity\n' +
      '   - Smart order routing\n' +
      '   - Fill processing\n' +
      '   - Latency budget: 50ms\n\n' +
      '3. **Position Tracker**:\n' +
      '   - Redis for real-time positions\n' +
      '   - Update on every fill\n' +
      '   - Latency budget: 10ms\n\n' +
      '4. **P&L Calculator**:\n' +
      '   - In-memory calculation\n' +
      '   - Real-time + EOD\n' +
      '   - Latency budget: 10ms\n\n' +
      '5. **Risk Manager**:\n' +
      '   - Pre-trade risk checks\n' +
      '   - Position/P&L limits\n' +
      '   - Latency budget: 10ms\n\n' +
      '**Total latency: 100ms ✓**',
    keyPoints: [
      'OMS: PostgreSQL persistence, REST API, order validation/routing (20ms budget)',
      'EMS: FIX engine (QuickFIX), broker connectivity, smart routing (50ms budget)',
      'Position Tracker: Redis real-time, update on fills (10ms budget)',
      'P&L Calculator: In-memory calculation, real-time + EOD (10ms budget)',
      'Risk Manager: Pre-trade checks, position/P&L limits (10ms budget), total latency 100ms',
    ],
  },
  {
    id: 'complete-trading-system-q-2',
    question:
      'You are tasked with migrating a legacy trading system (10 years old, monolithic, no tests) to a modern microservices architecture. What is your migration strategy?',
    sampleAnswer:
      'Legacy System Migration:\n\n' +
      '**Phase 1: Assessment (Month 1)**:\n' +
      '- Document current system\n' +
      '- Identify components: OMS, EMS, Risk, P&L\n' +
      '- Map dependencies\n' +
      '- Define microservices boundaries\n\n' +
      '**Phase 2: Strangler Pattern (Months 2-12)**:\n' +
      '- Build new services alongside legacy\n' +
      '- Route traffic gradually to new services\n' +
      '- Start with lowest-risk component (e.g., P&L calculator)\n' +
      '- Leave legacy running for 6 months\n\n' +
      '**Phase 3: Testing (Ongoing)**:\n' +
      "- Shadow mode: New system processes orders but doesn't execute\n" +
      '- Compare new vs legacy results\n' +
      '- Build comprehensive test suite\n\n' +
      '**Phase 4: Cutover (Month 12)**:\n' +
      '- Blue-green deployment\n' +
      '- Monitor closely for 2 weeks\n' +
      '- Keep legacy as backup\n\n' +
      '**Timeline: 12 months**',
    keyPoints: [
      'Assessment: Document current system, identify components, map dependencies (Month 1)',
      'Strangler pattern: Build new services alongside legacy, route traffic gradually, start with low-risk components',
      "Testing: Shadow mode (new system processes orders but doesn't execute), compare new vs legacy results",
      'Cutover: Blue-green deployment, monitor 2 weeks, keep legacy as backup (Month 12)',
      'Timeline: 12 months for complete migration, gradual rollout reduces risk',
    ],
  },
  {
    id: 'complete-trading-system-q-3',
    question:
      'What are the TOP 5 metrics you would track for a production trading system and why?',
    sampleAnswer:
      'Top 5 Trading System Metrics:\n\n' +
      '1. **Order Success Rate** (Target: >99%):\n' +
      '   - % of orders successfully executed\n' +
      '   - Most important: Directly measures system effectiveness\n' +
      '   - Alert if <95%\n\n' +
      '2. **Order Latency p99** (Target: <100ms):\n' +
      '   - 99th percentile order processing time\n' +
      '   - Impacts competitiveness (HFT)\n' +
      '   - Alert if >200ms\n\n' +
      '3. **Portfolio P&L** (Target: Track in real-time):\n' +
      '   - Realized + unrealized P&L\n' +
      '   - Risk management (alert if <-$50K)\n' +
      '   - Performance tracking\n\n' +
      '4. **Position Reconciliation Rate** (Target: >99.9%):\n' +
      '   - % of positions matching broker\n' +
      '   - Regulatory requirement\n' +
      '   - Alert if <99%\n\n' +
      '5. **System Uptime** (Target: >99.9%):\n' +
      '   - % of time system is operational\n' +
      '   - Business continuity\n' +
      '   - Alert if downtime >5 minutes',
    keyPoints: [
      'Order success rate (>99%): Most important, measures system effectiveness, alert if <95%',
      'Order latency p99 (<100ms): Impacts competitiveness (HFT), alert if >200ms',
      'Portfolio P&L: Real-time tracking, risk management (alert <-$50K), performance monitoring',
      'Position reconciliation (>99.9%): Regulatory requirement, alert if <99%',
      'System uptime (>99.9%): Business continuity, alert if downtime >5 minutes',
    ],
  },
];
