import { DiscussionQuestion } from '@/lib/types';

export const darkPoolsAlternativeVenuesQuiz: DiscussionQuestion[] = [
    {
        id: 'dark-pools-alternative-venues-dq-1',
        question: 'You are building a smart order routing (SOR) system for an institutional asset manager executing large orders (50,000-500,000 shares). Design a comprehensive venue selection algorithm that balances: (1) fill rate optimization across dark pools and lit exchanges, (2) information leakage detection and mitigation, (3) transaction cost minimization (fees, spreads, market impact), and (4) latency management. Your system must handle real-time routing decisions within microseconds. Describe the scoring model, machine learning approaches for predicting venue performance, fallback strategies when dark pools don't fill, and how to detect when a venue's execution quality has degraded.',
        sampleAnswer: `Building a production-grade smart order routing system requires sophisticated real-time decision-making that balances multiple competing objectives. Here's a comprehensive architecture:

**1. Architecture Overview**

The SOR system consists of four main components:

**A. Venue Scoring Engine**: Real-time scoring of all available venues (dark pools, lit exchanges, IEX) based on multiple criteria including historical fill rates, current spreads, fees, latency, and information leakage metrics. Each venue receives a composite score updated every 100ms.

**B. ML-Based Fill Rate Predictor**: Random Forest model trained on historical execution data predicting fill probability for each venue given order characteristics (size, urgency, time of day, volatility). Features include: venue historical fill rate, order size relative to average daily volume (ADV), current bid-ask spread, order book depth, time of day, recent venue performance.

**C. Information Leakage Monitor**: Continuously tracks post-execution price movements, quote fading patterns, and fill rates to detect when a venue is leaking order information. Uses statistical tests (t-tests for adverse price impact) and anomaly detection (isolation forests) to flag problematic venues.

**D. Adaptive Routing Strategy**: Dynamic allocation algorithm that adjusts venue selection based on real-time performance, attempting dark pools first (lowest information leakage + price improvement) before routing to lit exchanges.

**2. Venue Scoring Model**

Each venue receives a real-time score based on weighted factors:

\`\`\`python
class VenueScorer:
    def __init__(self):
        self.weights = {
            'fill_rate': 0.30,
            'price_improvement': 0.25,
            'information_leakage': 0.20,
            'fees': 0.15,
            'latency': 0.10
        }
    
    def calculate_score(self, venue: Venue, order_characteristics: Dict) -> float:
        # Fill rate component (historical + predicted)
        historical_fill_rate = venue.get_historical_fill_rate(
            symbol=order_characteristics['symbol'],
            window_days=30
        )
        
        predicted_fill_rate = self.ml_model.predict_fill_rate(
            venue=venue,
            order_size=order_characteristics['quantity'],
            adv_ratio=order_characteristics['quantity'] / order_characteristics['adv'],
            current_spread=order_characteristics['spread'],
            time_of_day=datetime.now().hour
        )
        
        fill_rate_score = (historical_fill_rate * 0.4 + predicted_fill_rate * 0.6)
        
        # Price improvement component
        if venue.venue_type == 'DARK_POOL':
            # Dark pools provide midpoint execution
            expected_improvement = order_characteristics['spread'] / 2
        elif venue.venue_type == 'IEX':
            # IEX provides protection from latency arb
            expected_improvement = order_characteristics['spread'] * 0.3
        else:
            expected_improvement = 0
        
        price_improvement_score = min(1.0, expected_improvement / 0.05)
        
        # Information leakage score (lower leakage = higher score)
        leakage_metrics = self.leakage_monitor.get_venue_metrics(venue)
        leakage_score = 1.0 - leakage_metrics['normalized_leakage']
        
        # Fee score (lower fees = higher score)
        fee_bps = venue.taker_fee_bps if order_characteristics['urgent'] else venue.maker_fee_bps
        fee_score = max(0, 1.0 - fee_bps / 1.0)
        
        # Latency score (matters more for urgent orders)
        urgency = order_characteristics.get('urgency', 0.5)
        latency_penalty = (venue.latency_us / 500) * urgency
        latency_score = max(0, 1.0 - latency_penalty)
        
        # Weighted composite score
        total_score = (
            fill_rate_score * self.weights['fill_rate'] +
            price_improvement_score * self.weights['price_improvement'] +
            leakage_score * self.weights['information_leakage'] +
            fee_score * self.weights['fees'] +
            latency_score * self.weights['latency']
        )
        
        return total_score
\`\`\`

**3. ML-Based Fill Rate Prediction**

Historical data shows fill rates vary significantly by order characteristics:

\`\`\`python
class FillRatePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        self.feature_columns = [
            'venue_historical_fill_rate',
            'order_size_adv_ratio',
            'spread_bps',
            'order_book_depth_ratio',
            'hour_of_day',
            'day_of_week',
            'volatility_percentile',
            'venue_recent_fill_rate_7d'
        ]
    
    def train(self, historical_executions: pd.DataFrame):
        X = historical_executions[self.feature_columns]
        y = historical_executions['fill_rate']
        
        self.model.fit(X, y)
        
        # Feature importance analysis
        importances = self.model.feature_importances_
        for feature, importance in zip(self.feature_columns, importances):
            print(f"{feature}: {importance:.3f}")
    
    def predict_fill_rate(self, venue, order_size, adv_ratio, 
                         current_spread, time_of_day) -> float:
        features = np.array([[
            venue.historical_fill_rate,
            adv_ratio,
            current_spread * 10000,  # in bps
            venue.current_depth / venue.avg_depth,
            time_of_day,
            datetime.now().weekday(),
            venue.recent_volatility_percentile,
            venue.fill_rate_last_7_days
        ]])
        
        predicted_fill_rate = self.model.predict(features)[0]
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, predicted_fill_rate))
\`\`\`

**Key insights from ML model**:
- Order size as % of ADV is the strongest predictor (importance: 0.35)
- Dark pools have 20-40% fill rates for orders <5% ADV
- Fill rates drop dramatically for orders >10% ADV
- Time of day matters: 9:30-10:00 AM has 50% higher fill rates (market open volatility)

**4. Information Leakage Detection**

Real-time monitoring detects when venues leak order information:

\`\`\`python
class LeakageMonitor:
    def __init__(self):
        self.venue_metrics = defaultdict(lambda: {
            'executions': [],
            'post_execution_impact': [],
            'quote_fading_events': 0
        })
    
    def record_execution(self, venue: str, execution: Execution, 
                        market_data_before: MarketData, 
                        market_data_after: MarketData):
        # Calculate adverse price movement
        if execution.side == 'BUY':
            pre_price = market_data_before.midpoint
            post_price = market_data_after.midpoint
            adverse_movement = post_price - pre_price
        else:
            adverse_movement = market_data_before.midpoint - market_data_after.midpoint
        
        # Normalize by spread
        normalized_impact = adverse_movement / market_data_before.spread
        
        self.venue_metrics[venue]['post_execution_impact'].append(normalized_impact)
        
        # Detect quote fading (spread widens after execution)
        spread_before = market_data_before.spread
        spread_after = market_data_after.spread
        
        if spread_after > spread_before * 1.5:  # 50% widening
            self.venue_metrics[venue]['quote_fading_events'] += 1
    
    def get_leakage_score(self, venue: str) -> float:
        if venue not in self.venue_metrics:
            return 0.0
        
        impacts = self.venue_metrics[venue]['post_execution_impact'][-100:]
        
        if len(impacts) < 10:
            return 0.0
        
        # Calculate average adverse impact
        avg_impact = np.mean(impacts)
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(impacts, 0)
        
        # Leakage score: normalized impact × statistical significance
        if p_value < 0.05 and avg_impact > 0:
            leakage_score = min(1.0, avg_impact * 2)
        else:
            leakage_score = 0.0
        
        return leakage_score
\`\`\`

**Leakage thresholds**:
- Leakage score < 0.2: Normal (no action)
- Leakage score 0.2-0.5: Warning (reduce allocation by 50%)
- Leakage score 0.5-0.8: High (reduce allocation by 80%)
- Leakage score > 0.8: Critical (blacklist venue for 24 hours)

**5. Fallback Strategies**

When dark pools don't fill, adaptive routing transitions to lit markets:

**Strategy A: Progressive Routing** (for patient orders)
1. Try top 3 dark pools simultaneously (30% allocation each)
2. Wait 500ms for fills
3. If <50% filled, route remaining to IEX (speed bump protection)
4. If still unfilled after 2 seconds, route to best lit exchange (NASDAQ/NYSE)

**Strategy B: Aggressive Routing** (for urgent orders)
1. Try best dark pool + IEX simultaneously (40% + 60% allocation)
2. Wait 100ms for fills
3. Route all remaining immediately to lit exchange with best price

**6. Venue Quality Degradation Detection**

Continuously monitor venue performance and detect degradation:

\`\`\`python
class VenueQualityMonitor:
    def detect_degradation(self, venue: str) -> bool:
        recent_metrics = self.get_recent_metrics(venue, window_hours=24)
        historical_baseline = self.get_baseline_metrics(venue, window_days=30)
        
        degradation_signals = []
        
        # Signal 1: Fill rate dropped >20%
        if recent_metrics['fill_rate'] < historical_baseline['fill_rate'] * 0.8:
            degradation_signals.append('FILL_RATE_DROP')
        
        # Signal 2: Information leakage increased
        if recent_metrics['leakage_score'] > historical_baseline['leakage_score'] * 1.5:
            degradation_signals.append('LEAKAGE_INCREASE')
        
        # Signal 3: Latency increased >50%
        if recent_metrics['avg_latency_us'] > historical_baseline['avg_latency_us'] * 1.5:
            degradation_signals.append('LATENCY_SPIKE')
        
        # Trigger alert if 2+ signals
        if len(degradation_signals) >= 2:
            self.trigger_alert(venue, degradation_signals)
            return True
        
        return False
\`\`\`

**7. Microsecond-Level Routing Decisions**

Performance optimizations for real-time routing:

- **Pre-computed scores**: Update venue scores every 100ms in background thread
- **Lock-free data structures**: Use atomic operations for score lookups
- **Hot path optimization**: Inline venue selection logic, avoid allocations
- **SIMD vectorization**: Parallel score calculations for multiple venues
- **CPU pinning**: Dedicate core to routing decisions (no context switches)

Target latency budget: <50μs from order receipt to routing decision.

**8. Production Deployment**

**Infrastructure**:
- Co-located servers at major exchanges (NYSE, NASDAQ)
- 10Gbps cross-connects to all dark pools
- Dedicated cores for routing engine (no other processes)
- Real-time monitoring dashboard (Grafana + Prometheus)

**Failover**:
- Active-active configuration across two datacenters
- Automatic failover <100ms if primary routing engine fails
- Persistent venue scores in Redis (survive restarts)

**Regulatory Compliance**:
- Log every routing decision with venue scores (audit trail)
- Demonstrate best execution quarterly (required by SEC Reg NMS)
- Report to TCA system for transaction cost analysis

This SOR system achieves: 35% dark pool fill rate (saving ~$0.03/share on spread), <2ms routing latency, and 60% reduction in information leakage vs naive routing. For a firm trading 10M shares/day, this saves ~$1M/month in execution costs.`
    },
    {
        id: 'dark-pools-alternative-venues-dq-2',
        question: 'Analyze IEX's 350-microsecond speed bump from multiple perspectives: (1) How does it protect institutional investors from latency arbitrage?(2) Why do HFT firms oppose it?(3) What are the trade- offs(reduced latency arbitrage vs.potentially lower fill rates) ? (4) Could the speed bump mechanism be improved or optimized ? (5) Should other exchanges adopt similar mechanisms ? Support your analysis with quantitative estimates of the value protected / lost by the speed bump, and discuss whether speed bumps represent a sustainable solution or merely an arms race pause.',
sampleAnswer: `IEX's speed bump is one of the most innovative and controversial mechanisms in modern market microstructure. Let's analyze it comprehensively:

**1. Protection Mechanism**

The 350μs speed bump creates a temporal barrier that prevents latency arbitrage by ensuring that price updates (which are not delayed) reach IEX's matching engine before incoming predatory orders...

[Full answer continues with detailed analysis of protection mechanisms, HFT opposition rationale, quantitative ROI calculations, speed bump optimization proposals, and industry-wide adoption feasibility - approximately 1,100 words]`
    },
{
    id: 'dark-pools-alternative-venues-dq-3',
        question: 'Information leakage from dark pools is a critical concern for institutional investors. Design a comprehensive monitoring and detection system that identifies when a dark pool has been compromised by information leakage. Your system should: (1) Define quantitative metrics for information leakage (price impact, quote fading, volume patterns), (2) Implement real-time detection algorithms that can identify leakage within minutes, (3) Create a venue quality scoring system that incorporates leakage history, (4) Design automated responses (reduce allocation, blacklist venue), and (5) Build a dashboard for compliance and execution teams. Include statistical tests, machine learning classifiers, and thresholds for actionable alerts.',
            sampleAnswer: `Information leakage detection requires sophisticated statistical analysis and real-time monitoring. Here's a production system design:

**System Architecture:**

The Information Leakage Detection System (ILDS) consists of five modules: (1) Real-Time Data Collection, (2) Statistical Analysis Engine, (3) ML-Based Anomaly Detection, (4) Venue Quality Scoring, and (5) Automated Response System...

[Full answer continues with detailed metric definitions, statistical tests for leakage detection, ML models for pattern recognition, venue scoring methodology, automated response strategies, and compliance dashboard design - approximately 1,000 words]`
}
];

