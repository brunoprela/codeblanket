import { MultipleChoiceQuestion } from '@/lib/types';

export const dataFeedProtocolsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'data-feed-protocols-mc-1',
    question:
      'You are parsing FIX messages and encounter this message: "8=FIX.4.4|9=100|35=8|39=2|54=1|55=AAPL|". The tag 39 has value "2". What does this mean?',
    options: [
      'Order was filled (executed)',
      'Order was partially filled',
      'Order was rejected',
      'Order is pending',
    ],
    correctAnswer: 0,
    explanation:
      'In FIX protocol, tag 39 is OrdStatus (Order Status). Value "2" means Filled (order completely executed). Common OrdStatus values: 0=New, 1=Partially Filled, 2=Filled, 4=Canceled, 6=Pending Cancel, 8=Rejected. This is an Execution Report (tag 35=8) indicating the order for AAPL with side Buy (tag 54=1) has been completely filled. Partially filled would be "1", Rejected would be "8", Pending would typically be "0" (New) or "A" (Pending New). Understanding FIX tag values is critical for order management systems - misinterpreting order status can lead to duplicate orders or missed fills. Production systems should have comprehensive FIX tag dictionaries and validation logic to catch invalid values.',
  },
  {
    id: 'data-feed-protocols-mc-2',
    question:
      'A FAST-encoded market data feed achieves 80% compression compared to text FIX. If uncompressed FIX messages average 200 bytes and you receive 100,000 messages per second, how much bandwidth does FAST save per day compared to text FIX?',
    options: [
      '~138 GB per day',
      '~691 GB per day',
      '~1.38 TB per day',
      '~6.91 TB per day',
    ],
    correctAnswer: 0,
    explanation:
      'Calculation: Text FIX: 100,000 msg/sec × 200 bytes = 20 MB/sec. FAST (80% compression): 20 MB/sec × 0.2 = 4 MB/sec. Savings: 20 - 4 = 16 MB/sec. Per day: 16 MB/sec × 86,400 seconds = 1,382,400 MB = 1,350 GB ≈ 138 GB after accounting for trading hours (6.5 hours/day). More precise: 16 MB/sec × 6.5 hours × 3,600 sec/hour = 374,400 MB ≈ 365 GB (full day) or 138 GB (trading hours). FAST compression is critical for high-frequency market data - at 1M msg/sec, you save 1.38 TB per trading day. This also reduces network latency (fewer bytes to transmit) and processing time. Cost savings: At $0.10/GB for data transfer, FAST saves $13.80/day = $3,500/year. For institutional feeds with 10M+ msg/sec, savings are 10× larger.',
  },
  {
    id: 'data-feed-protocols-mc-3',
    question:
      'Your NASDAQ ITCH processor receives an "Add Order" message (type A) with these fields: Order Ref=12345, Side=B, Shares=500, Stock=AAPL, Price=15025 (in 1/10000ths). Two messages later, you receive "Order Executed" (type E) with Order Ref=12345, Executed Shares=200. What should your order book show?',
    options: [
      'No order for ref 12345 (fully executed)',
      'Order ref 12345: 300 shares remaining at bid',
      "Order ref 12345: 500 shares at bid (execute doesn't change book)",
      'Order ref 12345: 200 shares executed, 300 shares cancelled',
    ],
    correctAnswer: 1,
    explanation:
      'Order book should show 300 shares remaining (500 original - 200 executed = 300). ITCH "Order Executed" messages indicate partial fills - the order remains in the book with reduced size. The order stays at the same price ($150.25, since 15025/10000 = 1.5025, which is $150.25). Only when cumulative executions equal original size, or a Delete/Cancel message arrives, should the order be removed. Common mistake: Removing order immediately on first execute (treats partial fill as full). Another mistake: Not tracking order references (order_ref uniquely identifies orders). Production order book must maintain order_ref → (price, remaining_size, side) mapping. When reconstructing NASDAQ order book from ITCH, expect 60-70% of orders to be partially executed multiple times before completion. This is critical for market making strategies that rely on accurate book depth.',
  },
  {
    id: 'data-feed-protocols-mc-4',
    question:
      'You are implementing FIX session management. The broker sends a Logon message (35=A) with MsgSeqNum (34) = 5, but your application expects sequence 1 (fresh session). What should you do?',
    options: [
      'Accept the logon with sequence 5 and continue from there',
      'Send a Logout and refuse the session (sequence mismatch)',
      'Send a Resend Request for messages 1-4 to fill the gap',
      'Reset your expected sequence to 5 and send a warning',
    ],
    correctAnswer: 2,
    explanation:
      "Send a Resend Request (MsgType 2) for sequences 1-4. FIX session management requires continuous sequence numbers - gaps indicate missed messages. The broker starting at sequence 5 suggests either: (1) They have state from a previous session, (2) They sent messages 1-4 before Logon (unusual), (3) Network issue caused missed messages. Requesting resend allows gap recovery. If broker responds with Gap Fill messages or actual messages 1-4, your sequence tracking synchronizes. If broker cannot resend (e.g., new session), they should respond with Gap Fill (MsgType 4) indicating sequences 1-4 don't exist. Never accept gaps silently - this violates FIX protocol and causes order management issues (missed fills, duplicate orders). Production systems maintain message store (persisted to disk) for exactly this purpose. Option A (accepting gap) risks missing important messages like execution reports. Option B (logout) is too aggressive - try recovery first. Option D (reset expectations) loses audit trail.",
  },
  {
    id: 'data-feed-protocols-mc-5',
    question:
      "A trading firm is choosing between direct exchange feeds (ITCH/FAST) vs vendor feeds (Bloomberg/Refinitiv) for market data. Direct feeds offer 1-10 μs latency but cost $300K setup + $50K/month. Vendor feeds offer 10-100 ms latency at $0 setup + $30K/month. The firm's fastest strategy has 5ms reaction time. Which should they choose?",
    options: [
      'Direct feeds - always choose lowest latency',
      "Vendor feeds - their strategy can't exploit sub-ms latency",
      'Direct feeds - the cost difference is negligible for a trading firm',
      'Vendor feeds initially, upgrade to direct feeds later if needed',
    ],
    correctAnswer: 1,
    explanation:
      "Vendor feeds are the correct choice. The firm's fastest strategy has 5ms (5,000 μs) reaction time, so sub-10μs data latency provides no benefit - the strategy cannot react faster than 5ms regardless. Direct feeds would save 90-99 μs (0.09-0.099 ms), which is 2% of the 5ms reaction time - negligible. Cost analysis: Direct feeds: $300K + ($50K × 12) = $900K first year. Vendor feeds: $0 + ($30K × 12) = $360K first year. Savings: $540K/year. This money is better spent on strategy research or infrastructure. Direct feeds only make sense for HFT strategies with < 100 μs reaction time, where every microsecond counts. At 5ms reaction time, the bottleneck is strategy logic, not data latency. Many firms waste money on ultra-low-latency feeds they cannot exploit. Correct approach: Start with vendor feeds, measure strategy performance, upgrade to direct feeds only if data latency is proven bottleneck (profile your system first). Exception: Market making strategies that need full order book depth may require ITCH regardless of latency.",
  },
];
