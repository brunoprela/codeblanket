import { QuizQuestion } from '@/lib/types';

export const realTimeAnalyticsDiscussionQuiz: QuizQuestion[] = [
  {
    id: 'rta-discussion-1',
    question:
      'Your company needs to build a real-time fraud detection system that analyzes 50,000 credit card transactions per second and flags suspicious activity within 100ms. The system must track: (1) unusual spending velocity per user, (2) geographic anomalies, (3) merchant patterns. Would you use exact computation or approximation algorithms, and why? Design the streaming pipeline architecture.',
    sampleAnswer: `**Recommendation: Use approximation algorithms for performance**

For 100ms latency at 50k TPS, exact computation is infeasible. Here\'s a comprehensive architecture:

**Streaming Pipeline:**
\`\`\`
Credit Card Transactions (50k/sec)
    ↓
  Kafka (buffer)
    ↓
Apache Flink (stateful processing)
    ↓
Redis (low-latency storage)
    ↓
Alert System
\`\`\`

**Flink Processing with Approximation:**

\`\`\`java
DataStream<Transaction> transactions = env
    .addSource (new KafkaSource<>("transactions"));

transactions
    .keyBy (t -> t.userId)
    .process (new FraudDetector())  // <100ms per event
    .addSink (new AlertSink());

class FraudDetector extends KeyedProcessFunction<String, Transaction, Alert> {
    // Approximation 1: HyperLogLog for unique merchants
    ValueState<HyperLogLog> uniqueMerchants;
    
    // Approximation 2: Count-Min Sketch for transaction counts  
    ValueState<CountMinSketch> merchantCounts;
    
    // Exact: Recent transactions (small window)
    ListState<Transaction> recentTxns;
    
    @Override
    public void processElement(Transaction t, Context ctx, Collector<Alert> out) {
        // Check 1: Spending velocity (exact, last 24h)
        List<Transaction> recent = recentTxns.get();
        double spent24h = recent.stream()
            .filter (tx -> tx.timestamp > now() - 24*HOUR)
            .mapToDouble (tx -> tx.amount)
            .sum();
            
        if (spent24h > 10000) {  // $10k in 24h
            out.collect (new Alert (t, "High velocity"));
        }
        
        // Check 2: Geographic anomaly (approximate)
        HyperLogLog hll = uniqueMerchants.value();
        hll.add (t.merchantCountry);
        if (hll.cardinality() > 5) {  // >5 countries in 24h
            out.collect (new Alert (t, "Geographic anomaly"));
        }
        
        // Check 3: Unusual merchant (approximate)
        CountMinSketch cms = merchantCounts.value();
        long merchantFreq = cms.estimate (t.merchantId);
        if (merchantFreq == 0) {  // First time at this merchant
            out.collect (new Alert (t, "New merchant"));
        }
        
        // Update state
        cms.add (t.merchantId);
        merchantCounts.update (cms);
    }
}
\`\`\`

**Why Approximation:**
- HyperLogLog: 12KB vs GBs for exact unique count
- Count-Min Sketch: 1MB vs potentially 100MB for exact counts
- Latency: <1ms per check vs 10-100ms for exact
- Result: <100ms total latency ✅

**Key Insight**: Perfect accuracy isn't needed for fraud detection. 2% error on "spent $9,800 vs $10,000" doesn't matter—both trigger review.`,
    keyPoints: [
      'Approximation algorithms (HyperLogLog, Count-Min Sketch) enable <100ms latency at scale',
      'Exact computation for critical recent data (last 24h transactions), approximate for patterns',
      'Stateful stream processing in Flink tracks per-user patterns efficiently',
      'Trade accuracy for speed: 2% error acceptable when detecting anomalies',
      'Memory efficiency: 12KB (HyperLogLog) vs GBs (exact unique counts)',
    ],
  },
  {
    id: 'rta-discussion-2',
    question:
      'Your real-time dashboard shows live metrics with 1-second refresh. You notice events arriving out-of-order by up to 30 seconds due to network delays. How would you handle late events using watermarks in Flink, and what trade-offs would you make between latency and completeness?',
    sampleAnswer: `**Watermark Strategy:**

\`\`\`java
// Allow 30 second lateness
WatermarkStrategy<Event> watermarkStrategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(30))
    .withIdleness(Duration.ofMinutes(1));

DataStream<Event> events = env
    .addSource (kafkaSource)
    .assignTimestampsAndWatermarks (watermarkStrategy);

events
    .keyBy (e -> e.userId)
    .window(TumblingEventTimeWindows.of(Time.seconds(60)))
    .allowedLateness(Time.seconds(30))  // Accept events up to 30s late
    .sideOutputLateData (lateEventsTag)  // Track dropped events
    .aggregate (new CountAggregator())
    .addSink (dashboardSink);
\`\`\`

**Trade-Off Analysis:**

| Lateness | Latency | Completeness | Recommendation |
|----------|---------|--------------|----------------|
| 0 sec | 1 sec | 70% (drops 30% late events) | Too aggressive |
| 30 sec | 31 sec | 99.9% | **Optimal** |
| 60 sec | 61 sec | 99.99% | Unnecessary delay |

**Result**: 30-second watermark balances 99.9% completeness with acceptable 31-second latency for dashboard refresh.`,
    keyPoints: [
      'Watermarks define how late events can arrive before being dropped',
      'allowedLateness extends window lifetime to accept late events',
      'Trade-off: Longer watermark = more complete but higher latency',
      'Side outputs capture dropped events for monitoring and correction',
      '30-second watermark captures 99.9% of events with acceptable latency',
    ],
  },
  {
    id: 'rta-discussion-3',
    question:
      'You need to display "top 10 trending hashtags" on a social media platform with 100,000 tweets per second. Exact counting requires storing all hashtags in memory (GBs). How would you use Count-Min Sketch to approximate top-K with bounded memory, and what accuracy guarantees can you provide?',
    sampleAnswer: `**Count-Min Sketch Implementation:**

\`\`\`python
class TopKHashtags:
    def __init__(self):
        # CMS: 10K width, 5 depth = 50K counters (200KB)
        self.cms = CountMinSketch (width=10000, depth=5)
        # Heap: Track top 100 candidates (1KB)
        self.topK = MinHeap(100)
        
    def process_tweet (self, hashtags):
        for tag in hashtags:
            # Increment count (O(1))
            count = self.cms.add (tag)
            
            # Update top-K if needed
            if count > self.topK.min():
                self.topK.add (tag, count)
    
    def get_top_k (self, k=10):
        return self.topK.top (k)
\`\`\`

**Memory Usage:**
- Exact: 1M unique hashtags × 8 bytes = 8MB (minimum)
- CMS: 50K counters × 4 bytes = 200KB (40x smaller!)

**Accuracy:**
- Never underestimates (monotonic)
- Overestimates by ε with probability δ
- ε = e / width = 2.718 / 10000 ≈ 0.0003 (0.03%)
- With proper tuning: 95%+ accuracy for top-10

**Trade-Off**: 200KB memory, 99% accuracy vs 8MB memory, 100% accuracy. For trending hashtags, 99% is sufficient—users don't notice if #10 and #11 swap positions.`,
    keyPoints: [
      'Count-Min Sketch provides approximate top-K with bounded memory (200KB vs 8MB)',
      'Never underestimates counts (monotonic property), only overestimates',
      'Error rate ε = e / width; larger width = better accuracy but more memory',
      "For top-K queries, approximation is acceptable—users don't need exact counts",
      'Real-world: 40x memory savings with 99% accuracy is excellent trade-off',
    ],
  },
];
