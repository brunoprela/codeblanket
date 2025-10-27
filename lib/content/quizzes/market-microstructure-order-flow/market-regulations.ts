import { DiscussionQuestion } from '@/lib/types';

export const marketRegulationsQuiz: DiscussionQuestion[] = [
    {
        id: 'market-regulations-dq-1',
        question: 'Design a comprehensive trade surveillance system that detects spoofing, layering, and wash trading in real-time. Your system must: (1) Define quantitative thresholds for suspicious patterns (order-to-trade ratios, cancel rates, time-to-cancel distributions), (2) Implement machine learning models to identify sophisticated manipulation strategies, (3) Generate alerts with different severity levels (low/medium/high/critical), (4) Integrate with order management systems to automatically pause suspicious accounts, and (5) Produce regulatory reports in SEC/FINRA format. Include specific algorithms, false positive mitigation strategies, and explain how to balance detection sensitivity with operational efficiency.',
        sampleAnswer: `A production-grade trade surveillance system requires sophisticated pattern detection combined with machine learning to identify market manipulation while minimizing false positives. Here's a comprehensive design:

**1. System Architecture**

The surveillance system monitors all trading activity in real-time across multiple dimensions:

**Core Components:**
- Event Stream Processor: Ingests order/trade events in real-time (Kafka, Flink)
- Pattern Detection Engine: Rule-based detection for known manipulation patterns
- ML Anomaly Detector: Identifies novel or sophisticated manipulation
- Alert Management System: Triages, scores, and routes alerts
- Case Management: Workflow for compliance officers to investigate alerts
- Regulatory Reporting: Auto-generates SEC/FINRA format reports

**2. Quantitative Thresholds for Pattern Detection**

**A. Spoofing Detection**

Spoofing: Placing orders with intent to cancel before execution to manipulate price.

**Thresholds:**
\`\`\`python
class SpoofingDetector:
    def __init__(self):
        self.thresholds = {
            'order_to_trade_ratio': 10.0,  # >10 orders per trade suspicious
            'large_order_cancel_rate': 0.80,  # >80% of large orders canceled
            'time_to_cancel_fast': 500,  # Orders canceled within 500ms
            'order_size_threshold': 10000,  # "Large" order = 10K+ shares
            'repeated_pattern_count': 5  # Same pattern 5+ times in hour
        }
    
    def detect_spoofing(self, trader_id: str, symbol: str, 
                       window_minutes: int = 60) -> Dict:
        """Detect spoofing patterns"""
        orders = self.get_orders(trader_id, symbol, window_minutes)
        trades = self.get_trades(trader_id, symbol, window_minutes)
        
        # Calculate metrics
        order_count = len(orders)
        trade_count = len(trades)
        order_to_trade = order_count / max(trade_count, 1)
        
        large_orders = [o for o in orders if o.quantity >= self.thresholds['order_size_threshold']]
        canceled_large = [o for o in large_orders if o.status == 'CANCELED']
        large_cancel_rate = len(canceled_large) / max(len(large_orders), 1)
        
        fast_cancels = [o for o in canceled_large 
                       if (o.cancel_time - o.place_time).total_seconds() * 1000 
                       < self.thresholds['time_to_cancel_fast']]
        
        # Spoofing score (0-100)
        score = 0
        if order_to_trade > self.thresholds['order_to_trade_ratio']:
            score += 30
        if large_cancel_rate > self.thresholds['large_order_cancel_rate']:
            score += 40
        if len(fast_cancels) > 3:
            score += 30
        
        return {
            'pattern': 'SPOOFING',
            'score': score,
            'severity': self._classify_severity(score),
            'metrics': {
                'order_to_trade_ratio': order_to_trade,
                'large_cancel_rate': large_cancel_rate,
                'fast_cancel_count': len(fast_cancels)
            },
            'evidence': fast_cancels[:10]  # Top 10 suspicious orders
        }
    
    def _classify_severity(self, score: int) -> str:
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
\`\`\`

**B. Layering Detection**

Layering: Placing multiple orders at different price levels to manipulate, then canceling.

**Pattern:**1. Place large order on one side (e.g., sell 10K @ $100.00)
2. Place many smaller orders on opposite side (buy 500 @ $99.95, $99.94, ... $99.80)
3. Cancel the "layers" once main order executes or price moves
4. Repeat pattern

**Thresholds:**
\`\`\`python
class LayeringDetector:
    def detect_layering(self, trader_id: str, symbol: str) -> Dict:
        """Detect layering patterns"""
        # Look for: many orders on one side, large order on opposite
        recent_orders = self.get_recent_orders(trader_id, symbol, seconds=60)
        
        buy_orders = [o for o in recent_orders if o.side == 'BUY']
        sell_orders = [o for o in recent_orders if o.side == 'SELL']
        
        # Check for imbalance (many on one side, few on other)
        if len(buy_orders) > 10 and len(sell_orders) < 3:
            # Potential layering: many buys, few sells
            large_sell = max(sell_orders, key=lambda o: o.quantity) if sell_orders else None
            
            if large_sell and large_sell.quantity > 5000:
                # Check if buy "layers" at multiple price levels
                unique_prices = len(set(o.price for o in buy_orders))
                
                if unique_prices >= 5:  # At least 5 different price levels
                    # Check cancel pattern
                    canceled = [o for o in buy_orders if o.status == 'CANCELED']
                    cancel_rate = len(canceled) / len(buy_orders)
                    
                    if cancel_rate > 0.70:  # 70%+ canceled
                        return {
                            'pattern': 'LAYERING',
                            'score': 75,
                            'severity': 'HIGH',
                            'main_order': large_sell.order_id,
                            'layer_count': len(buy_orders),
                            'price_levels': unique_prices,
                            'cancel_rate': cancel_rate
                        }
        
        # Symmetric check for opposite direction
        # ... (similar logic for many sells, few buys)
        
        return {'pattern': 'LAYERING', 'score': 0, 'severity': 'LOW'}
\`\`\`

**C. Wash Trading Detection**

Wash trading: Trading with yourself (or coordinated parties) to create false volume.

**Detection Methods:**1. **Account Matching**: Same trader on both sides
2. **Entity Matching**: Different accounts, same parent entity
3. **Pattern Matching**: Suspiciously simultaneous opposite orders
4. **Timing Analysis**: Orders placed within milliseconds, filled immediately

**Thresholds:**
\`\`\`python
class WashTradingDetector:
    def detect_wash_trades(self, trades: List[Trade]) -> Dict:
        """Detect wash trading patterns"""
        suspicious_trades = []
        
        for trade in trades:
            # Check 1: Same trader on both sides (direct wash trade)
            if trade.buyer_id == trade.seller_id:
                suspicious_trades.append({
                    'trade_id': trade.trade_id,
                    'type': 'SELF_TRADE',
                    'score': 100  # Definitive wash trade
                })
                continue
            
            # Check 2: Same parent entity (indirect wash trade)
            buyer_entity = self.get_parent_entity(trade.buyer_id)
            seller_entity = self.get_parent_entity(trade.seller_id)
            
            if buyer_entity == seller_entity:
                suspicious_trades.append({
                    'trade_id': trade.trade_id,
                    'type': 'ENTITY_WASH_TRADE',
                    'score': 90
                })
                continue
            
            # Check 3: Timing pattern (orders placed within 10ms, filled immediately)
            buyer_order = self.get_order(trade.buy_order_id)
            seller_order = self.get_order(trade.sell_order_id)
            
            time_diff_ms = abs((buyer_order.timestamp - seller_order.timestamp).total_seconds() * 1000)
            
            if time_diff_ms < 10:  # Within 10ms
                # Further check: Are these traders connected?
                connection_score = self.check_trader_connection(trade.buyer_id, trade.seller_id)
                
                if connection_score > 0.7:  # High likelihood of coordination
                    suspicious_trades.append({
                        'trade_id': trade.trade_id,
                        'type': 'COORDINATED_WASH_TRADE',
                        'score': 70,
                        'time_diff_ms': time_diff_ms,
                        'connection_score': connection_score
                    })
        
        return {
            'pattern': 'WASH_TRADING',
            'suspicious_count': len(suspicious_trades),
            'trades': suspicious_trades
        }
    
    def check_trader_connection(self, trader1: str, trader2: str) -> float:
        """Check if two traders are likely coordinating"""
        # Look for patterns indicating connection:
        # - Frequent trading with each other (>50% of volume)
        # - Always on opposite sides
        # - No other trading partners
        # - Same IP address / geolocation
        # - Similar trading patterns
        
        historical_trades = self.get_trader_pair_history(trader1, trader2, days=30)
        
        if not historical_trades:
            return 0.0
        
        trader1_total_volume = self.get_total_volume(trader1, days=30)
        pair_volume = sum(t.quantity for t in historical_trades)
        volume_concentration = pair_volume / trader1_total_volume
        
        # High concentration suggests coordination
        return min(1.0, volume_concentration)
\`\`\`

**3. Machine Learning for Sophisticated Manipulation**

ML models detect novel patterns that evade rule-based systems:

**Feature Engineering:**
\`\`\`python
class ManipulationMLModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=200, max_depth=6)
        self.features = [
            # Order flow features
            'order_to_trade_ratio',
            'cancel_rate',
            'modify_rate',
            'avg_time_to_cancel_ms',
            'order_size_std_dev',
            
            # Timing features
            'orders_per_second_peak',
            'order_clustering_score',  # How clustered in time
            'night_trading_ratio',  # Trading at odd hours
            
            # Price impact features
            'avg_price_impact_bps',
            'price_reversal_rate',  # Price reverts after trader's orders
            
            # Network features
            'unique_counterparties',
            'counterparty_concentration',  # Trade with same parties repeatedly
            
            # Behavioral features
            'trading_days_active',
            'symbols_traded_count',
            'consistent_pattern_score'  # Same behavior every day
        ]
    
    def train(self, historical_data: pd.DataFrame, labels: pd.Series):
        """Train on historical manipulat events"""
        X = historical_data[self.features]
        y = labels  # 1 = manipulation, 0 = legitimate
        
        # Handle class imbalance (manipulation is rare)
        from imblearn.over_sampling import SMOTE
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        
        self.model.fit(X_resampled, y_resampled)
        
        # Feature importance
        importances = self.model.feature_importances_
        for feature, importance in sorted(zip(self.features, importances), 
                                         key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.3f}")
    
    def predict_manipulation_probability(self, trader_activity: Dict) -> float:
        """Predict probability of manipulation"""
        features = np.array([[trader_activity[f] for f in self.features]])
        prob = self.model.predict_proba(features)[0][1]  # Probability of class 1 (manipulation)
        return prob
\`\`\`

**4. Alert System with Severity Levels**

**Alert Scoring & Prioritization:**
\`\`\`python
class AlertManager:
    def generate_alert(self, detection_result: Dict) -> Dict:
        """Generate alert with severity and priority"""
        base_score = detection_result['score']
        
        # Adjust score based on trader history
        trader_id = detection_result.get('trader_id')
        trader_risk_profile = self.get_trader_risk_profile(trader_id)
        
        if trader_risk_profile['prior_violations'] > 0:
            base_score += 20  # Increase severity for repeat offenders
        
        if trader_risk_profile['trading_volume_percentile'] > 90:
            base_score += 10  # Large traders get more scrutiny
        
        # Classify severity
        if base_score >= 80:
            severity = 'CRITICAL'
            action = 'AUTO_PAUSE_TRADING'
            sla_hours = 1
        elif base_score >= 60:
            severity = 'HIGH'
            action = 'IMMEDIATE_REVIEW'
            sla_hours = 4
        elif base_score >= 40:
            severity = 'MEDIUM'
            action = 'REVIEW_WITHIN_24H'
            sla_hours = 24
        else:
            severity = 'LOW'
            action = 'WEEKLY_BATCH_REVIEW'
            sla_hours = 168
        
        alert = {
            'alert_id': self.generate_alert_id(),
            'timestamp': datetime.now(),
            'trader_id': trader_id,
            'pattern': detection_result['pattern'],
            'score': base_score,
            'severity': severity,
            'recommended_action': action,
            'sla_hours': sla_hours,
            'evidence': detection_result.get('evidence', []),
            'assigned_to': self.assign_to_analyst(severity)
        }
        
        # Auto-actions for critical alerts
        if severity == 'CRITICAL':
            self.auto_pause_trader(trader_id)
            self.send_urgent_notification(alert)
        
        return alert
    
    def auto_pause_trader(self, trader_id: str):
        """Automatically pause trading for suspicious account"""
        # Integrate with Order Management System
        self.oms.set_trader_status(trader_id, status='PAUSED')
        self.oms.cancel_all_orders(trader_id)
        
        # Log action
        self.log_action({
            'action': 'AUTO_PAUSE',
            'trader_id': trader_id,
            'timestamp': datetime.now(),
            'reason': 'Critical manipulation alert'
        })
\`\`\`

**5. Regulatory Reporting (SEC/FINRA Format)**

**Suspicious Activity Report (SAR) Generation:**
\`\`\`python
class RegulatoryReporter:
    def generate_sar(self, alert: Dict, investigation_notes: str) -> Dict:
        """Generate Suspicious Activity Report for SEC/FINRA"""
        sar = {
            'filing_institution': 'XYZ Brokerage',
            'filing_date': datetime.now().strftime('%Y-%m-%d'),
            
            # Part I: Subject Information
            'subject': {
                'account_number': alert['trader_id'],
                'name': self.get_trader_name(alert['trader_id']),
                'tax_id': self.get_trader_tin(alert['trader_id']),
                'address': self.get_trader_address(alert['trader_id'])
            },
            
            # Part II: Suspicious Activity Information
            'activity': {
                'type': self._map_pattern_to_sar_type(alert['pattern']),
                'date_range': {
                    'start': alert['timestamp'] - timedelta(days=30),
                    'end': alert['timestamp']
                },
                'product_type': 'Equities',
                'instruments': self.get_traded_symbols(alert['trader_id']),
                'total_dollar_amount': self.calculate_suspicious_volume_dollars(alert)
            },
            
            # Part III: Narrative
            'narrative': self._generate_narrative(alert, investigation_notes),
            
            # Part IV: Filing Institution Contact
            'contact': {
                'name': 'Chief Compliance Officer',
                'phone': '555-0100',
                'email': 'compliance@xyzbrokerage.com'
            }
        }
        
        # Submit to FinCEN (Financial Crimes Enforcement Network)
        self.submit_to_fincen(sar)
        
        return sar
    
    def _generate_narrative(self, alert: Dict, investigation_notes: str) -> str:
        """Generate narrative description of suspicious activity"""
        pattern = alert['pattern']
        trader = alert['trader_id']
        
        narrative = f"""
        Subject: Suspected {pattern} by account {trader}
        
        Our automated surveillance system flagged suspicious trading activity consistent 
        with {pattern.lower()} manipulation. The system detected the following patterns:
        
        {self._format_evidence(alert['evidence'])}
        
        Investigation Notes:
        {investigation_notes}
        
        Recommended Action: {alert['recommended_action']}
        
        This activity is being reported to regulators in accordance with applicable 
        securities laws and regulations.
        """
        
        return narrative.strip()
\`\`\`

**6. False Positive Mitigation**

**Strategies to Reduce False Positives:**1. **Contextual Filtering:**
   - Exclude market making registered firms (higher order-to-trade ratios expected)
   - Adjust thresholds during high volatility (more cancels are normal)
   - Consider asset class (options have higher cancel rates than equities)

2. **Pattern Confirmation:**
   - Require multiple indicators to fire (not just one threshold)
   - Look for repeated patterns over time (not isolated incidents)
   - Use ensemble of detectors (rule-based + ML agree)

3. **Human-in-the-Loop:**
   - Auto-dismiss LOW severity alerts after ML review
   - MEDIUM alerts require analyst review
   - HIGH/CRITICAL alerts require senior compliance officer approval

4. **Feedback Loop:**
   - Track false positive rate per detector
   - Adjust thresholds based on analyst feedback
   - Retrain ML models monthly with labeled data

**7. Balancing Sensitivity vs. Operational Efficiency**

**Challenge**: Too sensitive → overwhelmed with false positives. Too lenient → miss real manipulation.

**Solution: Risk-Based Tiering:**

**Tier 1 (High-Risk Traders):** 5% of traders
- High volume, prior violations, or suspicious patterns
- Strictest thresholds, real-time monitoring
- Every alert reviewed within 4 hours

**Tier 2 (Medium-Risk):** 25% of traders
- Moderate volume, no violations
- Standard thresholds, daily batch review
- Alerts reviewed within 24 hours

**Tier 3 (Low-Risk):** 70% of traders
- Low volume, long history, no issues
- Lenient thresholds, weekly batch review
- Only HIGH/CRITICAL alerts escalated

**Operational Metrics:**
- Target: <5% false positive rate
- Analyst capacity: 20 alerts/day per analyst
- SLA: 95% of alerts reviewed within SLA window

This system successfully balances regulatory compliance, operational efficiency, and detection effectiveness by combining rule-based detection with ML, tiered monitoring, and automated actions for critical cases.`
    },
    {
        id: 'market-regulations-dq-2',
        question: 'Implement a Reg NMS Rule 611 (Order Protection Rule) compliance engine that prevents trade-throughs. Your system receives quotes from 13 exchanges and must: (1) Calculate NBBO in real-time (<100 microseconds latency), (2) Validate every order execution against NBBO before completion, (3) Handle edge cases (stale quotes, exchange outages, ISOs), (4) Generate exception reports for regulators, and (5) Maintain audit logs proving compliance. Describe the data structures for efficient NBBO calculation, race condition handling when quotes update during execution, and how to prove to auditors that no trade-through violations occurred.',
        sampleAnswer: `Reg NMS Rule 611 compliance requires microsecond-level NBBO tracking with bulletproof auditability. Here's a production-grade implementation:

**1. System Architecture**

**Core Requirements:**
- Receive quotes from 13+ exchanges (NYSE, NASDAQ, Cboe, IEX, ARCA, BATS, EDGX, EDGA, PSX, CHX, NSX, LTSE, MEMX)
- Calculate NBBO in <100μs
- Validate executions against NBBO before completion
- Handle 100,000+ quotes/second
- Zero trade-through violations (strict compliance)

**Architecture:**
\`\`\`
┌─────────────────┐
│ Exchange Feeds  │ (13+ exchanges via SIP or direct feeds)
└────────┬────────┘
         │ Quote updates
         v
┌─────────────────┐
│ NBBO Calculator │ (Lock-free, <100μs latency)
└────────┬────────┘
         │ Best bid/ask
         v
┌─────────────────┐
│ Order Validator │ (Pre-execution check)
└────────┬────────┘
         │ Approved orders
         v
┌─────────────────┐
│ Execution Engine│ (Route to best venue)
└────────┬────────┘
         │ Executions
         v
┌─────────────────┐
│ Audit Logger    │ (Immutable trail)
└─────────────────┘
\`\`\`

**2. Data Structures for Efficient NBBO Calculation**

**Challenge**: Need O(1) access to best bid/ask across 13 exchanges, updated millions of times per second.

**Solution: Lock-Free Exchange Quote Map + Cached NBBO**

\`\`\`python
from dataclasses import dataclass
from typing import Dict, Optional
import threading
from collections import defaultdict

@dataclass
class ExchangeQuote:
    """Quote from a single exchange"""
    exchange: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp_ns: int
    sequence: int  # Exchange sequence number

class NBBOCalculator:
    """
    Lock-free NBBO calculator using atomic updates.
    
    Key insights:
    - Each exchange quote stored separately (13 slots)
    - NBBO cached and recalculated only when quotes change
    - Use versioning to detect concurrent updates
    """
    
    def __init__(self):
        # Exchange quotes (one per exchange)
        self.exchange_quotes: Dict[str, ExchangeQuote] = {}
        
        # Cached NBBO (updated atomically)
        self.nbbo_cache = {
            'bid': None,
            'ask': None,
            'bid_size': 0,
            'ask_size': 0,
            'bid_exchanges': [],
            'ask_exchanges': [],
            'timestamp_ns': 0,
            'version': 0  # Incremented on each update
        }
        
        # Lock for NBBO updates (coarse-grained, only during recalculation)
        self.nbbo_lock = threading.Lock()
        
        # Performance metrics
        self.update_count = 0
        self.recalc_count = 0
    
    def update_exchange_quote(self, quote: ExchangeQuote):
        """
        Update quote from one exchange and recalculate NBBO if needed.
        
        Time complexity: O(N) where N = number of exchanges (13)
        Typical latency: 10-50 microseconds
        """
        old_quote = self.exchange_quotes.get(quote.exchange)
        self.exchange_quotes[quote.exchange] = quote
        self.update_count += 1
        
        # Determine if NBBO needs recalculation
        needs_recalc = False
        
        if not old_quote:
            needs_recalc = True  # New exchange
        else:
            # Check if this quote could affect NBBO
            old_best_bid = self.nbbo_cache['bid']
            old_best_ask = self.nbbo_cache['ask']
            
            # Bid changed and could be new best bid
            if quote.bid != old_quote.bid and (old_best_bid is None or 
                                               quote.bid >= old_best_bid or 
                                               old_quote.bid == old_best_bid):
                needs_recalc = True
            
            # Ask changed and could be new best ask
            if quote.ask != old_quote.ask and (old_best_ask is None or 
                                               quote.ask <= old_best_ask or 
                                               old_quote.ask == old_best_ask):
                needs_recalc = True
        
        if needs_recalc:
            self._recalculate_nbbo(quote.timestamp_ns)
    
    def _recalculate_nbbo(self, timestamp_ns: int):
        """Recalculate NBBO from all exchange quotes"""
        with self.nbbo_lock:
            self.recalc_count += 1
            
            # Find best bid (highest bid across all exchanges)
            best_bid = None
            best_bid_size = 0
            bid_exchanges = []
            
            for exchange, quote in self.exchange_quotes.items():
                if quote.bid is not None:
                    if best_bid is None or quote.bid > best_bid:
                        best_bid = quote.bid
                        best_bid_size = quote.bid_size
                        bid_exchanges = [exchange]
                    elif quote.bid == best_bid:
                        # Multiple exchanges at best bid
                        best_bid_size += quote.bid_size
                        bid_exchanges.append(exchange)
            
            # Find best ask (lowest ask across all exchanges)
            best_ask = None
            best_ask_size = 0
            ask_exchanges = []
            
            for exchange, quote in self.exchange_quotes.items():
                if quote.ask is not None:
                    if best_ask is None or quote.ask < best_ask:
                        best_ask = quote.ask
                        best_ask_size = quote.ask_size
                        ask_exchanges = [exchange]
                    elif quote.ask == best_ask:
                        best_ask_size += quote.ask_size
                        ask_exchanges.append(exchange)
            
            # Update cached NBBO atomically
            self.nbbo_cache = {
                'bid': best_bid,
                'ask': best_ask,
                'bid_size': best_bid_size,
                'ask_size': best_ask_size,
                'bid_exchanges': bid_exchanges,
                'ask_exchanges': ask_exchanges,
                'timestamp_ns': timestamp_ns,
                'version': self.nbbo_cache['version'] + 1
            }
    
    def get_nbbo(self) -> Dict:
        """
        Get current NBBO (O(1) read from cache).
        
        Returns snapshot of NBBO with version number.
        """
        return self.nbbo_cache.copy()
    
    def get_nbbo_versioned(self) -> tuple:
        """Get NBBO with version (for detecting concurrent updates)"""
        nbbo = self.nbbo_cache.copy()
        return nbbo, nbbo['version']

**3. Order Validation Against NBBO**

**Pre-Execution Check (CRITICAL PATH):**

\`\`\`python
class OrderValidator:
    """Validate orders against NBBO to prevent trade-throughs"""
    
    def __init__(self, nbbo_calculator: NBBOCalculator):
        self.nbbo_calc = nbbo_calculator
        self.violations_prevented = 0
    
    def validate_order(self, order: Dict) -> Dict:
        """
        Validate order execution against current NBBO.
        
        Returns: {
            'approved': bool,
            'reason': str,
            'nbbo_snapshot': dict,
            'nbbo_version': int
        }
        """
        # Get NBBO snapshot with version
        nbbo, version = self.nbbo_calc.get_nbbo_versioned()
        
        side = order['side']
        execution_price = order['price']
        order_type = order['type']
        
        # Handle different order types
        if order_type == 'MARKET':
            # Market orders must route to NBBO venue
            approved = True  # Market orders always approved (routed to best venue)
            reason = "Market order - will route to NBBO venue"
        
        elif order_type == 'LIMIT':
            # Check for trade-through
            if side == 'BUY':
                # Buy order: execution price must not exceed best ask
                if nbbo['ask'] is None:
                    approved = True
                    reason = "No offers available"
                elif execution_price <= nbbo['ask']:
                    approved = True
                    reason = "Execution at or better than NBBO ask"
                else:
                    approved = False
                    reason = f"TRADE-THROUGH: Buy @ \${execution_price:.2f} > NBBO ask \${nbbo['ask']:.2f}"
                    self.violations_prevented += 1
            
            else:  # SELL
                # Sell order: execution price must not be below best bid
                if nbbo['bid'] is None:
                    approved = True
                    reason = "No bids available"
                elif execution_price >= nbbo['bid']:
                    approved = True
                    reason = "Execution at or better than NBBO bid"
                else:
                    approved = False
                    reason = f"TRADE-THROUGH: Sell @ \${execution_price:.2f} < NBBO bid \${nbbo['bid']:.2f}"
                    self.violations_prevented += 1
        
        elif order_type == 'ISO':  # Intermarket Sweep Order
            # ISO orders are exempt from trade-through rule
            # (firm simultaneously sweeps all better-priced venues)
            approved = True
            reason = "ISO order - exempt from Rule 611"
        
        return {
            'approved': approved,
            'reason': reason,
            'nbbo_snapshot': nbbo,
            'nbbo_version': version,
            'timestamp_ns': time.perf_counter_ns()
        }

**4. Handling Edge Cases**

**A. Race Conditions (Quote Updates During Execution)**

**Problem**: NBBO changes between validation and execution.

**Solution: Optimistic Validation with Retry**

\`\`\`python
class ExecutionEngine:
    def execute_order(self, order: Dict, max_retries: int = 3) -> Dict:
        """Execute order with retry on NBBO change"""
        
        for attempt in range(max_retries):
            # Validate against current NBBO
            validation = self.validator.validate_order(order)
            
            if not validation['approved']:
                return {
                    'status': 'REJECTED',
                    'reason': validation['reason'],
                    'nbbo_at_validation': validation['nbbo_snapshot']
                }
            
            # Attempt execution
            nbbo_version_at_validation = validation['nbbo_version']
            
            # Send order to exchange/venue
            execution_result = self.send_to_venue(order, validation['nbbo_snapshot'])
            
            # Check if NBBO changed during execution
            current_nbbo, current_version = self.nbbo_calc.get_nbbo_versioned()
            
            if current_version == nbbo_version_at_validation:
                # NBBO unchanged - execution is valid
                return {
                    'status': 'EXECUTED',
                    'execution': execution_result,
                    'nbbo_at_execution': current_nbbo
                }
            else:
                # NBBO changed - validate execution against new NBBO
                if self._revalidate_execution(execution_result, current_nbbo):
                    return {
                        'status': 'EXECUTED',
                        'execution': execution_result,
                        'nbbo_at_execution': current_nbbo,
                        'note': 'NBBO changed during execution but still compliant'
                    }
                else:
                    # Execution would violate current NBBO - cancel and retry
                    self.cancel_execution(execution_result)
                    # Retry with new NBBO
                    continue
        
        # Max retries exceeded
        return {
            'status': 'REJECTED',
            'reason': 'Max retries exceeded due to NBBO volatility'
        }

**B. Stale Quotes**

**Problem**: Exchange feed delayed or disconnected.

**Solution: Quote Timeout + Exchange Status Tracking**

\`\`\`python
class ExchangeHealthMonitor:
    def __init__(self):
        self.exchange_status = {}  # exchange -> {last_update, status}
        self.quote_timeout_ms = 1000  # 1 second timeout
    
    def mark_quote_received(self, exchange: str):
        """Mark that exchange sent quote (is healthy)"""
        self.exchange_status[exchange] = {
            'last_update': time.time(),
            'status': 'ACTIVE'
        }
    
    def check_exchange_health(self, exchange: str) -> str:
        """Check if exchange is healthy or stale"""
        if exchange not in self.exchange_status:
            return 'UNKNOWN'
        
        time_since_update = (time.time() - self.exchange_status[exchange]['last_update']) * 1000
        
        if time_since_update > self.quote_timeout_ms:
            self.exchange_status[exchange]['status'] = 'STALE'
            return 'STALE'
        else:
            return 'ACTIVE'
    
    def get_active_exchanges(self) -> List[str]:
        """Return list of exchanges with current quotes"""
        return [ex for ex in self.exchange_status 
                if self.check_exchange_health(ex) == 'ACTIVE']

# Modified NBBO calculation excludes stale exchanges
def _recalculate_nbbo_with_health_check(self):
    active_exchanges = self.health_monitor.get_active_exchanges()
    
    # Only consider quotes from active exchanges
    for exchange, quote in self.exchange_quotes.items():
        if exchange in active_exchanges:
            # ... (normal NBBO calculation)

**C. Exchange Outages**

**Problem**: Major exchange goes offline (e.g., NYSE outage).

**Solution: Exclude Offline Exchanges from NBBO**

\`\`\`python
# When exchange outage detected:
def handle_exchange_outage(self, exchange: str):
    """Handle exchange outage (exclude from NBBO)"""
    # Remove exchange quotes
    if exchange in self.exchange_quotes:
        del self.exchange_quotes[exchange]
    
    # Mark exchange as offline
    self.exchange_status[exchange] = 'OFFLINE'
    
    # Recalculate NBBO without this exchange
    self._recalculate_nbbo(time.perf_counter_ns())
    
    # Alert compliance team
    self.alert_compliance(f"Exchange {exchange} offline - excluded from NBBO")

**5. Audit Logging (Immutable Trail)**

**Requirements:**
- Log every quote update
- Log every NBBO calculation
- Log every order validation decision
- Tamper-proof (write-once)
- Fast retrieval for audits

**Implementation:**

\`\`\`python
class AuditLogger:
    """Immutable audit trail for Reg NMS compliance"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.current_file = None
        self.sequence = 0
    
    def log_quote_update(self, exchange: str, quote: ExchangeQuote):
        """Log quote update from exchange"""
        self._write_log_entry({
            'event_type': 'QUOTE_UPDATE',
            'sequence': self._next_sequence(),
            'timestamp_ns': quote.timestamp_ns,
            'exchange': exchange,
            'bid': quote.bid,
            'ask': quote.ask,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size
        })
    
    def log_nbbo_calculation(self, nbbo: Dict):
        """Log NBBO calculation result"""
        self._write_log_entry({
            'event_type': 'NBBO_CALC',
            'sequence': self._next_sequence(),
            'timestamp_ns': nbbo['timestamp_ns'],
            'bid': nbbo['bid'],
            'ask': nbbo['ask'],
            'bid_exchanges': nbbo['bid_exchanges'],
            'ask_exchanges': nbbo['ask_exchanges'],
            'version': nbbo['version']
        })
    
    def log_order_validation(self, order_id: str, validation: Dict):
        """Log order validation decision"""
        self._write_log_entry({
            'event_type': 'ORDER_VALIDATION',
            'sequence': self._next_sequence(),
            'timestamp_ns': validation['timestamp_ns'],
            'order_id': order_id,
            'approved': validation['approved'],
            'reason': validation['reason'],
            'nbbo_version': validation['nbbo_version'],
            'nbbo_snapshot': validation['nbbo_snapshot']
        })
    
    def _write_log_entry(self, entry: Dict):
        """Write entry to immutable log file"""
        # JSON format for easy parsing
        log_line = json.dumps(entry) + '\\n'
        
        # Write to append-only file
        with open(self._get_current_log_file(), 'a') as f:
            f.write(log_line)
        
        # Optionally write to database
        self.db.insert('audit_log', entry)
    
    def _next_sequence(self) -> int:
        """Get next sequence number (monotonically increasing)"""
        self.sequence += 1
        return self.sequence

**6. Proving Compliance to Auditors**

**Audit Query Examples:**

\`\`\`python
class ComplianceAuditor:
    """Tools for auditors to verify compliance"""
    
    def verify_no_trade_throughs(self, start_date: datetime, end_date: datetime) -> Dict:
        """Verify no trade-through violations occurred"""
        
        # Query all executions in date range
        executions = self.db.query("""
            SELECT order_id, side, execution_price, execution_time, nbbo_version
            FROM executions
            WHERE execution_time BETWEEN ? AND ?
        """, (start_date, end_date))
        
        violations = []
        
        for execution in executions:
            # Retrieve NBBO at time of execution
            nbbo = self.db.query("""
                SELECT bid, ask, bid_exchanges, ask_exchanges
                FROM nbbo_log
                WHERE version = ?
            """, (execution['nbbo_version'],))
            
            # Check for violation
            if execution['side'] == 'BUY':
                if execution['execution_price'] > nbbo['ask']:
                    violations.append({
                        'order_id': execution['order_id'],
                        'execution_price': execution['execution_price'],
                        'nbbo_ask': nbbo['ask'],
                        'violation_amount': execution['execution_price'] - nbbo['ask']
                    })
            else:  # SELL
                if execution['execution_price'] < nbbo['bid']:
                    violations.append({
                        'order_id': execution['order_id'],
                        'execution_price': execution['execution_price'],
                        'nbbo_bid': nbbo['bid'],
                        'violation_amount': nbbo['bid'] - execution['execution_price']
                    })
        
        return {
            'total_executions': len(executions),
            'violations': violations,
            'violation_count': len(violations),
            'compliance_rate': 100 * (1 - len(violations) / max(len(executions), 1))
        }
    
    def generate_compliance_report(self, month: str) -> str:
        """Generate monthly compliance report for regulators"""
        result = self.verify_no_trade_throughs(
            start_date=datetime.strptime(f"{month}-01", "%Y-%m-%d"),
            end_date=datetime.strptime(f"{month}-01", "%Y-%m-%d") + timedelta(days=32)
        )
        
        report = f"""
        REG NMS RULE 611 COMPLIANCE REPORT
        Period: {month}
        
        Total Orders Executed: {result['total_executions']:,}
        Trade-Through Violations: {result['violation_count']}
        Compliance Rate: {result['compliance_rate']:.4f}%
        
        NBBO Calculation Performance:
        - Average latency: {self.get_avg_nbbo_latency()}μs
        - 99th percentile latency: {self.get_p99_nbbo_latency()}μs
        - Quote updates processed: {self.get_quote_count():,}
        
        All audit logs available in database table `audit_log`.
        No material violations detected.
        
        Compliance Officer: [Name]
        Date: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        return report

This implementation achieves <100μs NBBO calculation, zero trade-throughs through rigorous validation, and provides complete audit trail for regulatory compliance.`
    },
    {
        id: 'market-regulations-dq-3',
        question: 'Your European trading firm must comply with MiFID II algorithmic trading requirements. Design a comprehensive compliance program that includes: (1) Algorithmic trading registration (determining if >50% threshold is met), (2) Best execution reporting with quarterly statistics, (3) Transaction reporting with microsecond timestamps, (4) Kill switch implementation that can halt all algos within 1 second, and (5) Double volume cap monitoring for dark pools (8% per venue, 4% total). Explain how to prove best execution to regulators, the technical architecture for microsecond-accurate timestamps, and strategies for staying within dark pool volume caps.',
        sampleAnswer: `MiFID II compliance is significantly more stringent than US regulations, requiring comprehensive monitoring and reporting systems. Here's a complete compliance program:

**1. Algorithmic Trading Registration (>50% Threshold)**

**MiFID II Definition:**
A firm is considered an "algorithmic trader" if >50% of orders are generated by algorithms.

**Threshold Calculation:**
\`\`\`python
class AlgoTradingThresholdMonitor:
    """Monitor if firm exceeds 50% algorithmic trading threshold"""
    
    def __init__(self):
        self.orders_by_type = {'ALGO': 0, 'MANUAL': 0}
        self.reset_date = datetime.now().date()
    
    def record_order(self, order_id: str, is_algorithmic: bool):
        """Record order as algorithmic or manual"""
        # Reset counters daily
        if datetime.now().date() > self.reset_date:
            self.orders_by_type = {'ALGO': 0, 'MANUAL': 0}
            self.reset_date = datetime.now().date()
        
        if is_algorithmic:
            self.orders_by_type['ALGO'] += 1
        else:
            self.orders_by_type['MANUAL'] += 1
    
    def get_algo_percentage(self) -> float:
        """Calculate percentage of algorithmic orders"""
        total = self.orders_by_type['ALGO'] + self.orders_by_type['MANUAL']
        if total == 0:
            return 0.0
        return (self.orders_by_type['ALGO'] / total) * 100
    
    def is_registration_required(self) -> bool:
        """Check if MiFID II registration required"""
        # Measure over trailing 30 days
        trailing_30d_algo_pct = self.calculate_trailing_percentage(days=30)
        
        if trailing_30d_algo_pct > 50.0:
            return True
        else:
            return False
    
    def calculate_trailing_percentage(self, days: int) -> float:
        """Calculate algo percentage over trailing period"""
        start_date = datetime.now() - timedelta(days=days)
        
        orders = self.db.query("""
            SELECT is_algorithmic, COUNT(*) as count
            FROM orders
            WHERE timestamp >= ?
            GROUP BY is_algorithmic
        """, (start_date,))
        
        algo_count = next((o['count'] for o in orders if o['is_algorithmic']), 0)
        total_count = sum(o['count'] for o in orders)
        
        return (algo_count / total_count * 100) if total_count > 0 else 0.0

**Registration Process:**
- Submit notification to national competent authority (FCA in UK, BaFin in Germany, etc.)
- Provide algorithm descriptions
- Detail risk controls and testing procedures
- Ongoing: Update registration when algorithms change materially

**2. Best Execution Reporting (Quarterly)**

**MiFID II Requirement:**
Firms must publish quarterly reports demonstrating best execution across execution venues.

**Best Execution Metrics:**
\`\`\`python
class BestExecutionReporter:
    """Generate MiFID II best execution reports"""
    
    def __init__(self):
        self.execution_venues = ['NYSE', 'NASDAQ', 'LSE', 'Euronext', 'Deutsche Boerse', 
                                'Sigma X', 'Turquoise', 'IEX']
    
    def generate_quarterly_report(self, quarter: str) -> Dict:
        """Generate RTS 27 best execution report"""
        
        report = {
            'period': quarter,
            'asset_classes': {}
        }
        
        # Analyze by asset class (equities, bonds, derivatives, etc.)
        for asset_class in ['EQUITIES', 'BONDS', 'DERIVATIVES']:
            executions = self.get_executions(quarter, asset_class)
            
            venue_stats = self.calculate_venue_statistics(executions)
            
            report['asset_classes'][asset_class] = {
                'total_executions': len(executions),
                'total_volume': sum(e['quantity'] for e in executions),
                'venues': venue_stats,
                'best_execution_factors': self.analyze_best_execution_factors(executions)
            }
        
        return report
    
    def calculate_venue_statistics(self, executions: List[Dict]) -> Dict:
        """Calculate statistics per venue"""
        venue_stats = defaultdict(lambda: {
            'execution_count': 0,
            'volume': 0,
            'avg_price_improvement_bps': 0,
            'avg_speed_ms': 0,
            'fill_rate': 0
        })
        
        for execution in executions:
            venue = execution['venue']
            venue_stats[venue]['execution_count'] += 1
            venue_stats[venue]['volume'] += execution['quantity']
            
            # Price improvement vs. NBBO
            if execution['side'] == 'BUY':
                improvement = execution['nbbo_ask'] - execution['execution_price']
            else:
                improvement = execution['execution_price'] - execution['nbbo_bid']
            
            improvement_bps = (improvement / execution['execution_price']) * 10000
            venue_stats[venue]['avg_price_improvement_bps'] += improvement_bps
            
            # Speed (time from order to execution)
            speed_ms = (execution['execution_time'] - execution['order_time']).total_seconds() * 1000
            venue_stats[venue]['avg_speed_ms'] += speed_ms
        
        # Calculate averages
        for venue in venue_stats:
            count = venue_stats[venue]['execution_count']
            venue_stats[venue]['avg_price_improvement_bps'] /= count
            venue_stats[venue]['avg_speed_ms'] /= count
        
        return dict(venue_stats)
    
    def analyze_best_execution_factors(self, executions: List[Dict]) -> Dict:
        """Analyze factors considered for best execution"""
        return {
            'price': {
                'weight': 0.50,
                'description': 'Execution price relative to NBBO',
                'avg_improvement_bps': self._calculate_avg_price_improvement(executions)
            },
            'costs': {
                'weight': 0.20,
                'description': 'Transaction fees and commissions',
                'avg_cost_bps': self._calculate_avg_costs(executions)
            },
            'speed': {
                'weight': 0.15,
                'description': 'Time to execution',
                'avg_speed_ms': self._calculate_avg_speed(executions)
            },
            'likelihood_of_execution': {
                'weight': 0.10,
                'description': 'Fill rate',
                'fill_rate': self._calculate_fill_rate(executions)
            },
            'size': {
                'weight': 0.05,
                'description': 'Ability to execute full size',
                'avg_fill_ratio': self._calculate_avg_fill_ratio(executions)
            }
        }
    
    def publish_report(self, report: Dict):
        """Publish report on firm website (MiFID II requirement)"""
        # Generate PDF/HTML report
        html_report = self._generate_html_report(report)
        
        # Publish to firm website
        self.upload_to_website(html_report, f"best_execution_{report['period']}.html")
        
        # Submit to regulator
        self.submit_to_regulator(report)

**3. Transaction Reporting with Microsecond Timestamps**

**MiFID II Requirement:**
Report all transactions to regulators within T+1 with microsecond timestamp accuracy.

**Timestamp Synchronization Architecture:**
\`\`\`python
class MiFIDTimestampSystem:
    """Microsecond-accurate timestamps using PTP"""
    
    def __init__(self):
        # Use PTP (Precision Time Protocol) for clock sync
        self.ptp_synchronized = self.verify_ptp_sync()
        self.max_allowed_offset_us = 100  # 100μs max offset from UTC
    
    def verify_ptp_sync(self) -> bool:
        """Verify PTP synchronization meets MiFID II requirements"""
        # Check PTP daemon status
        import subprocess
        result = subprocess.run(['pmc', '-u', '-b', '0', 'GET', 'TIME_STATUS_NP'],
                              capture_output=True, text=True)
        
        # Parse offset from UTC
        for line in result.stdout.split('\\n'):
            if 'master_offset' in line:
                offset_ns = int(line.split()[-1])
                offset_us = offset_ns / 1000
                
                if abs(offset_us) <= self.max_allowed_offset_us:
                    return True
                else:
                    self.alert_compliance(f"PTP offset {offset_us}μs exceeds threshold")
                    return False
        
        return False
    
    def get_regulatory_timestamp(self) -> str:
        """Get timestamp in MiFID II format (microsecond precision)"""
        # ISO 8601 format with microseconds
        # Example: 2025-10-26T10:30:45.123456Z
        now = datetime.utcnow()
        return now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    def generate_transaction_report(self, execution: Dict) -> Dict:
        """Generate MiFID II transaction report (ARM format)"""
        return {
            # Identification
            'transaction_reference_number': execution['execution_id'],
            'trading_venue': execution['venue'],
            'reporting_entity': 'XYZ Trading Ltd',
            'lei': 'LEI123456789',  # Legal Entity Identifier
            
            # Instrument
            'instrument_isin': execution['isin'],
            'instrument_name': execution['symbol'],
            
            # Transaction details
            'buy_sell_indicator': execution['side'],
            'quantity': execution['quantity'],
            'price': execution['price'],
            'price_currency': 'EUR',
            'notional_amount': execution['quantity'] * execution['price'],
            
            # Timestamp (microsecond precision)
            'transaction_timestamp': self.get_regulatory_timestamp(),
            
            # Client information
            'client_id_type': 'LEI',
            'client_id': execution['client_lei'],
            'client_country': execution['client_country'],
            
            # Algorithm flag
            'algo_indicator': 'ALGO' if execution['is_algorithmic'] else 'NONE',
            'algo_id': execution.get('algorithm_id', 'N/A')
        }
    
    def submit_to_arm(self, reports: List[Dict]):
        """Submit transaction reports to Approved Reporting Mechanism (ARM)"""
        # Format reports in FIX or ISO 20022 XML
        xml_report = self._format_as_iso20022_xml(reports)
        
        # Submit to ARM (e.g., Approved Reporting Mechanism like Cappitech, Bloomberg, etc.)
        response = requests.post(
            'https://arm-provider.com/api/submit',
            data=xml_report,
            headers={'Content-Type': 'application/xml'},
            auth=('username', 'password')
        )
        
        if response.status_code == 200:
            self.log_submission(reports, 'SUCCESS')
        else:
            self.log_submission(reports, 'FAILED')
            self.alert_compliance('Transaction report submission failed')

**4. Kill Switch (Halt All Algos Within 1 Second)**

**MiFID II Requirement:**
Ability to cancel all outstanding orders and halt all algorithmic trading within 1 second.

**Kill Switch Implementation:**
\`\`\`python
class MiFIDKillSwitch:
    """Emergency kill switch for all algorithmic trading"""
    
    def __init__(self):
        self.kill_switch_active = False
        self.algorithms = []  # List of all algorithm instances
        self.oms = OrderManagementSystem()
    
    def activate_kill_switch(self, reason: str, activated_by: str):
        """Activate kill switch - halt all trading within 1 second"""
        activation_time = time.perf_counter()
        
        print(f"[KILL SWITCH] ACTIVATED by {activated_by}: {reason}")
        self.kill_switch_active = True
        
        # Step 1: Cancel all outstanding orders (parallel)
        cancel_threads = []
        for exchange in self.oms.get_active_exchanges():
            thread = threading.Thread(target=self._cancel_all_orders_on_exchange, 
                                     args=(exchange,))
            thread.start()
            cancel_threads.append(thread)
        
        # Wait for all cancels to complete (with timeout)
        for thread in cancel_threads:
            thread.join(timeout=0.5)  # 500ms max wait
        
        # Step 2: Halt all algorithms
        for algorithm in self.algorithms:
            algorithm.halt()
        
        # Step 3: Disconnect from exchanges (prevent new orders)
        self.oms.disconnect_all()
        
        elapsed_ms = (time.perf_counter() - activation_time) * 1000
        
        # Log activation
        self.log_kill_switch_activation({
            'timestamp': datetime.now(),
            'reason': reason,
            'activated_by': activated_by,
            'elapsed_ms': elapsed_ms,
            'orders_canceled': self.oms.get_canceled_count(),
            'algos_halted': len(self.algorithms)
        })
        
        # Alert regulators if required
        if reason == 'REGULATORY_REQUEST':
            self.notify_regulator()
        
        print(f"[KILL SWITCH] Complete in {elapsed_ms:.0f}ms")
        
        return {'status': 'ACTIVATED', 'elapsed_ms': elapsed_ms}
    
    def _cancel_all_orders_on_exchange(self, exchange: str):
        """Cancel all orders on specific exchange"""
        orders = self.oms.get_active_orders(exchange)
        
        for order in orders:
            try:
                self.oms.cancel_order(order['order_id'])
            except Exception as e:
                print(f"Error canceling {order['order_id']}: {e}")
    
    def test_kill_switch(self):
        """Test kill switch in simulation mode"""
        # MiFID II requires periodic testing
        print("[KILL SWITCH TEST] Starting test...")
        
        # Simulate activation without actual cancellation
        test_start = time.perf_counter()
        
        # Test each component
        self._test_order_cancellation()
        self._test_algo_halt()
        self._test_connectivity_shutdown()
        
        test_duration_ms = (time.perf_counter() - test_start) * 1000
        
        result = {
            'test_date': datetime.now(),
            'test_duration_ms': test_duration_ms,
            'passed': test_duration_ms < 1000,  # Must complete in <1 second
            'components_tested': 3
        }
        
        self.log_kill_switch_test(result)
        
        return result

**Physical Kill Switch:**
\`\`\`python
# GPIO integration for physical button
import RPi.GPIO as GPIO

class PhysicalKillSwitchButton:
    """Physical button that activates kill switch"""
    
    def __init__(self, gpio_pin: int, kill_switch: MiFIDKillSwitch):
        self.gpio_pin = gpio_pin
        self.kill_switch = kill_switch
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Register interrupt
        GPIO.add_event_detect(gpio_pin, GPIO.FALLING, 
                            callback=self._button_pressed, bouncetime=300)
    
    def _button_pressed(self, channel):
        """Callback when physical button pressed"""
        print("Physical kill switch button pressed!")
        self.kill_switch.activate_kill_switch(
            reason="PHYSICAL_BUTTON",
            activated_by="TRADING_DESK"
        )

**5. Dark Pool Volume Caps (8% per venue, 4% total)**

**MiFID II Requirement:**
Dark trading cannot exceed 8% of total volume per venue, or 4% across all venues, for any given stock.

**Volume Cap Monitoring:**
\`\`\`python
class DarkPoolVolumeCapMonitor:
    """Monitor MiFID II dark pool volume caps"""
    
    def __init__(self):
        self.per_venue_cap = 0.08  # 8%
        self.total_cap = 0.04  # 4%
        self.lookback_days = 30
    
    def calculate_dark_volume_percentage(self, symbol: str, venue: str = None) -> Dict:
        """Calculate dark trading percentage"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        if venue:
            # Per-venue calculation
            dark_volume = self.get_dark_volume(symbol, venue, start_date, end_date)
            total_volume = self.get_total_volume(symbol, start_date, end_date)
            
            percentage = (dark_volume / total_volume * 100) if total_volume > 0 else 0
            
            return {
                'symbol': symbol,
                'venue': venue,
                'dark_volume': dark_volume,
                'total_volume': total_volume,
                'dark_percentage': percentage,
                'cap': self.per_venue_cap * 100,
                'breach': percentage > (self.per_venue_cap * 100),
                'margin': (self.per_venue_cap * 100) - percentage
            }
        else:
            # Total across all venues
            dark_volume_all = self.get_total_dark_volume(symbol, start_date, end_date)
            total_volume = self.get_total_volume(symbol, start_date, end_date)
            
            percentage = (dark_volume_all / total_volume * 100) if total_volume > 0 else 0
            
            return {
                'symbol': symbol,
                'venue': 'ALL',
                'dark_volume': dark_volume_all,
                'total_volume': total_volume,
                'dark_percentage': percentage,
                'cap': self.total_cap * 100,
                'breach': percentage > (self.total_cap * 100),
                'margin': (self.total_cap * 100) - percentage
            }
    
    def monitor_and_enforce(self, symbol: str):
        """Monitor caps and enforce restrictions"""
        # Check per-venue caps
        for venue in ['Turquoise', 'Sigma X', 'Aquis', 'ITG POSIT']:
            result = self.calculate_dark_volume_percentage(symbol, venue)
            
            if result['breach']:
                self.handle_breach(symbol, venue, result)
            elif result['margin'] < 1.0:  # Within 1% of cap
                self.send_warning(symbol, venue, result)
        
        # Check total cap
        total_result = self.calculate_dark_volume_percentage(symbol, venue=None)
        
        if total_result['breach']:
            self.handle_total_breach(symbol, total_result)
    
    def handle_breach(self, symbol: str, venue: str, result: Dict):
        """Handle volume cap breach"""
        print(f"[VOLUME CAP BREACH] {symbol} on {venue}: {result['dark_percentage']:.2f}% > {result['cap']:.2f}%")
        
        # MiFID II action: Suspend dark trading on this venue for 6 months
        self.suspend_dark_trading(symbol, venue, months=6)
        
        # Notify regulator
        self.notify_regulator_of_breach(symbol, venue, result)
    
    def suspend_dark_trading(self, symbol: str, venue: str, months: int):
        """Suspend dark trading per MiFID II requirement"""
        suspension = {
            'symbol': symbol,
            'venue': venue,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=months*30),
            'reason': 'MiFID II volume cap breach'
        }
        
        # Add to suspension list
        self.db.insert('dark_pool_suspensions', suspension)
        
        # Configure routing system to avoid this venue/symbol combination
        self.routing_system.add_venue_symbol_exclusion(venue, symbol)
    
    def get_allowed_dark_volume(self, symbol: str, venue: str) -> int:
        """Calculate remaining dark volume allowed before hitting cap"""
        result = self.calculate_dark_volume_percentage(symbol, venue)
        
        # Calculate shares remaining before cap
        cap_volume = result['total_volume'] * self.per_venue_cap
        remaining = cap_volume - result['dark_volume']
        
        return max(0, int(remaining))

**Staying Within Caps (Proactive Strategy):**
\`\`\`python
class DarkPoolRoutingStrategy:
    """Smart routing that respects volume caps"""
    
    def route_order(self, symbol: str, quantity: int) -> str:
        """Route order while respecting dark pool caps"""
        
        # Check allowed volume for each dark pool
        venue_allowances = {}
        for venue in self.dark_pools:
            allowed = self.volume_monitor.get_allowed_dark_volume(symbol, venue)
            venue_allowances[venue] = allowed
        
        # Sort by allowance (descending)
        sorted_venues = sorted(venue_allowances.items(), key=lambda x: x[1], reverse=True)
        
        # Route to venue with most headroom
        for venue, allowance in sorted_venues:
            if allowance >= quantity:
                return venue
        
        # If no dark pool has capacity, route to lit exchange
        return 'NYSE' # Or best lit venue

This comprehensive MiFID II compliance program ensures regulatory adherence through automated monitoring, microsecond timestamps, emergency controls, and proactive volume cap management.`
    }
];
