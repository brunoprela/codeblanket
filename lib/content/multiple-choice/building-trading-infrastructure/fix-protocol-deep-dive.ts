export const fixProtocolDeepDiveMC = [
  {
    id: 'fix-protocol-deep-dive-mc-1',
    question:
      'What is the purpose of the "checksum" field (tag 10) in FIX protocol messages?',
    options: [
      'To encrypt the message for security',
      'To ensure message integrity by verifying no corruption occurred during transmission',
      'To identify the message type',
      'To track the sequence number',
    ],
    correctAnswer: 1,
    explanation:
      'FIX Checksum Purpose: INTEGRITY CHECK to detect transmission errors (bit flips, corruption). CALCULATION: Sum of ASCII values of all characters (from BeginString through end of message, excluding checksum field itself) modulo 256. FORMAT: 3-digit padded string (e.g., "010", "123", "256" becomes "000"). EXAMPLE: Message: "8=FIX.4.2\\x019=50\\x0135=0\\x01", ASCII values: 56+61+70+73+88+46+52+46+50+1+57+61+53+48+1+51+53+61+48+1 = 1234 (example), Checksum: 1234 % 256 = 210, Result: "10=210\\x01". VERIFICATION: Receiver recalculates checksum, compares with received value, if mismatch → reject message (corruption detected). NOT SECURITY: Checksum does NOT encrypt or authenticate (anyone can calculate correct checksum). Not for tampering protection (use encryption + HMAC for security). NOT UNIQUE IDENTIFIER: Tag 35 (MsgType) identifies message type, tag 34 (MsgSeqNum) tracks sequence. Checksum only validates integrity. REAL-WORLD: All FIX messages must have valid checksum, reject on mismatch (FIX specification requirement). Critical for reliable trading (corrupted price = catastrophic loss).',
  },
  {
    id: 'fix-protocol-deep-dive-mc-2',
    question:
      'What is the purpose of sequence numbers (MsgSeqNum, tag 34) in FIX protocol?',
    options: [
      'To identify the type of message being sent',
      'To ensure message ordering and detect lost/duplicate messages, enabling reliable communication',
      'To encrypt messages for security',
      'To calculate the message checksum',
    ],
    correctAnswer: 1,
    explanation:
      "FIX Sequence Numbers: RELIABLE ORDERING and LOSS DETECTION in unreliable network. MECHANISM: Outgoing: Start at 1, increment for EVERY message (including heartbeats), never skip or reuse. Incoming: Expect sequential numbers (1, 2, 3, ...), detect gaps or duplicates. GAP DETECTION: Receive seq 15 but expect 10 → gap of 5 messages. Send ResendRequest (35=2): BeginMsgSeqNo=10, EndMsgSeqNo=14. Broker resends missing messages with PossDupFlag=Y (tag 43). DUPLICATE DETECTION: Receive seq 8 but expect 10 (already processed 8) → ignore duplicate. Broker may resend after reconnect. EXAMPLE: Session flow: Send: Logon (seq 1) → NewOrder (seq 2) → Heartbeat (seq 3). Receive: Logon (seq 1) → ExecReport (seq 2) → Heartbeat (seq 3). Network issue: Send seq 2, broker doesn't receive. Broker sends seq 2, we send seq 3. Broker receives seq 3, expects 2 → sends ResendRequest for 2. We resend seq 2 with PossDupFlag=Y. SESSION PERSISTENCE: On disconnect, both sides remember last seq numbers. Reconnect: Continue from last seq (e.g., send 101 if last was 100). Or reset to 1 if new trading day (ResetSeqNumFlag=Y in Logon). WHY CRITICAL: Lost order = missed trade opportunity, lost fill = position error, duplicate order = double execution (trade twice accidentally). Sequence numbers prevent all these catastrophic failures. ALTERNATIVES: TCP guarantees delivery but not application-level ordering (FIX adds application-layer reliability). Without sequence numbers: No way to detect lost messages, no way to detect duplicates, no way to resynchronize after disconnect.",
  },
  {
    id: 'fix-protocol-deep-dive-mc-3',
    question:
      'What is the purpose of "heartbeat" messages (MsgType=0) in FIX protocol?',
    options: [
      'To send order updates to the exchange',
      'To maintain connection and detect dead connections by exchanging periodic keep-alive messages',
      'To request market data from the broker',
      'To authenticate the session',
    ],
    correctAnswer: 1,
    explanation:
      'FIX Heartbeat Purpose: DETECT DEAD CONNECTIONS (network failure, crash, freeze). MECHANISM: During Logon (35=A), negotiate HeartBtInt=30 (heartbeat every 30 seconds). If no message sent in 30s → send Heartbeat (35=0). If no message received in 60s (2× interval) → connection dead. TEST REQUEST: If 1.5× interval (45s) with no message → send TestRequest (35=1) with TestReqID=123. Expect Heartbeat (35=0) with TestReqID=123 within 10s. If no response → disconnect. EXAMPLE: Normal flow: 10:00:00 - Send order, 10:00:15 - Receive fill, 10:00:30 - No activity for 15s, still OK, 10:00:45 - No activity for 45s → send TestRequest, 10:00:47 - Receive Heartbeat → connection alive, 10:01:00 - No activity for 60s → send Heartbeat. Dead connection: 10:00:00 - Send order, 10:00:15 - Network cable unplugged, 10:00:45 - No receive for 45s → TestRequest (no response), 10:01:00 - No receive for 60s → DISCONNECT, reconnect. WHY IMPORTANT: Avoid "zombie connections" (TCP connected but application frozen). Example: Broker application crashes but TCP stays up → no orders execute, heartbeat detects within 60s → reconnect → resume trading. Without heartbeats: Could wait minutes/hours before realizing connection dead, miss critical market opportunities. AUTOMATIC: Most FIX engines handle heartbeats automatically (QuickFIX, etc.), application doesn\'t need to manually send. BANDWIDTH: Heartbeats are small (~50 bytes), negligible bandwidth (<1KB/minute), essential reliability mechanism.',
  },
  {
    id: 'fix-protocol-deep-dive-mc-4',
    question:
      'What is the difference between ClOrdID (tag 11) and OrderID (tag 37) in FIX protocol?',
    options: [
      'They are the same field with different names',
      'ClOrdID is client-assigned unique identifier, OrderID is broker/exchange-assigned identifier',
      'ClOrdID is for market orders, OrderID is for limit orders',
      'OrderID is encrypted, ClOrdID is plain text',
    ],
    correctAnswer: 1,
    explanation:
      'ClOrdID vs OrderID: ClOrdID (Tag 11 - Client Order ID): Assigned by CLIENT when creating order, must be UNIQUE across all client orders (never reuse), format: client choice (e.g., "ORD-12345", "STRAT1-001"), purpose: track order through lifecycle (cancels, replaces reference original ClOrdID), client uses this to match fills to original order. OrderID (Tag 37 - Order ID): Assigned by BROKER/EXCHANGE when order accepted, unique in broker\'s system (not client\'s responsibility), format: broker choice (e.g., "12345678"), purpose: broker\'s internal tracking, regulatory reporting. EXAMPLE FLOW: Client creates: ClOrdID="ORD-12345", sends NewOrderSingle (35=D) with ClOrdID=ORD-12345. Broker accepts: Assigns OrderID="87654321", sends ExecutionReport with ClOrdID=ORD-12345 (echo) + OrderID=87654321. Fills reference BOTH: ExecReport: ClOrdID=ORD-12345 (so client knows which order) + OrderID=87654321 (broker tracking). Client cancels: Sends OrderCancelRequest with OrigClOrdID=ORD-12345 (original) + ClOrdID=ORD-12345-C1 (new cancel ID). Broker may use OrderID internally but returns ClOrdID to client. WHY TWO IDs: CLIENT needs consistent ID (ClOrdID) across session restarts, broker reconnects. BROKER needs internal ID (OrderID) for routing, database, regulatory reporting. Independence: Client can restart, generate new ClOrdIDs, doesn\'t need to know broker\'s OrderID scheme. ANALOGY: ClOrdID = your reference number, OrderID = broker\'s confirmation number. Like tracking package: Your order # (ClOrdID) vs FedEx tracking # (OrderID). CRITICAL: Always use ClOrdID in client code (you control it), store ClOrdID → OrderID mapping for reconciliation.',
  },
  {
    id: 'fix-protocol-deep-dive-mc-5',
    question:
      'What does ExecType (tag 150) indicate in a FIX Execution Report?',
    options: [
      'The type of security being traded (stock, option, future)',
      'The reason for the execution report: New order (0), Trade/Fill (F), Canceled (4), Rejected (8), etc.',
      'The exchange where the order was executed',
      'The time the order was executed',
    ],
    correctAnswer: 1,
    explanation:
      'FIX ExecType (Tag 150): REASON for ExecutionReport message. COMMON VALUES: 0 = New: Order accepted (entered into system), OrdStatus = 0 (New). 1 = Partial Fill: Some quantity filled (but not complete), OrdStatus = 1 (PartiallyFilled). 2 = Fill: Complete fill, OrdStatus = 2 (Filled). 4 = Canceled: Order canceled, OrdStatus = 4 (Canceled). 5 = Replace: Order modified, OrdStatus varies. 8 = Rejected: Order rejected (validation failed), OrdStatus = 8 (Rejected). F = Trade: Same as Fill (some systems use F instead of 2), indicates execution occurred. EXAMPLE LIFECYCLE: Send NewOrderSingle: ClOrdID=ORD-001, Symbol=AAPL, Qty=100, Price=$150. Receive ExecReport #1: ExecType=0 (New), OrdStatus=0, CumQty=0, LeavesQty=100 → order accepted. Receive ExecReport #2: ExecType=F (Trade), OrdStatus=1 (PartiallyFilled), LastQty=50, LastPx=$150.00, CumQty=50, LeavesQty=50 → partial fill. Receive ExecReport #3: ExecType=F (Trade), OrdStatus=2 (Filled), LastQty=50, LastPx=$150.01, CumQty=100, LeavesQty=0 → complete. WHY IMPORTANT: ExecType tells you WHAT HAPPENED (new, fill, cancel, reject), OrdStatus tells you CURRENT STATE (new, partial, filled, canceled), must check BOTH: ExecType=F + OrdStatus=1 → partial fill (more coming), ExecType=F + OrdStatus=2 → final fill (done). REJECTION EXAMPLE: Send order: Price=$150.00, outside allowed range. Receive ExecReport: ExecType=8 (Rejected), OrdStatus=8 (Rejected), Text="Price outside limit up/down" (tag 58) → order failed. PRACTICAL: On each ExecutionReport: Check ExecType to know event type, check OrdStatus for current state, check CumQty for total filled so far, check LeavesQty for remaining.',
  },
];
