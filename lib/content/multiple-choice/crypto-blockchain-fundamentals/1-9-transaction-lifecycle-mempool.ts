import { MultipleChoiceQuestion } from '@/lib/types';

export const transactionLifecycleMempoolMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tlm-mc-1',
      question: 'What is the "mempool" in Bitcoin?',
      options: [
        'A pool of miners competing for blocks',
        'Memory pool of unconfirmed transactions waiting to be mined',
        'A backup of the blockchain',
        'Storage for old transactions',
      ],
      correctAnswer: 1,
      explanation:
        "Mempool: Each node's pool of valid but unconfirmed transactions. When transaction broadcast: (1) Nodes validate, (2) Add to their mempool, (3) Relay to peers. Miners select transactions from mempool to include in blocks (typically highest fees first). After confirmation, removed from mempool. Key: Each node has OWN mempool (not global). During congestion, mempools grow large (100K+ transactions). Purge policy: Transactions eventually dropped if not confirmed (typically after 2 weeks).",
    },
    {
      id: 'tlm-mc-2',
      question: "How does EIP-1559 change Ethereum's fee market?",
      options: [
        'It eliminates transaction fees',
        'It introduces base fee (burned) + priority tip, with dynamic base fee adjustment',
        'It makes all transactions free',
        'It adds a fixed fee of 1559 gwei',
      ],
      correctAnswer: 1,
      explanation:
        'EIP-1559 (London hard fork, 2021): Two-part fee structure. Base fee: Algorithmically adjusted based on block fullness, burned (reduces ETH supply). Priority tip: User-set tip to miner for prioritization. If block >50% full: base fee increases. If <50% full: decreases. Benefits: (1) Predictable fees (wallets estimate base fee), (2) Deflationary pressure (burn), (3) Reduced overpayment. Users still bid priority tips during congestion. Pre-EIP-1559: Pure auction, unpredictable fees, all goes to miners.',
    },
    {
      id: 'tlm-mc-3',
      question: 'What is Replace-by-Fee (RBF) in Bitcoin?',
      options: [
        'Refunding transaction fees',
        'Replacing unconfirmed transaction with higher-fee version',
        'Miners returning fees',
        'Fee estimation algorithm',
      ],
      correctAnswer: 1,
      explanation:
        'RBF (BIP 125): Replace unconfirmed transaction with new version paying higher fee. Use case: Transaction stuck in mempool with low fee, bump fee to get confirmed faster. Process: (1) Broadcast transaction A with low fee, (2) Transaction A stuck, (3) Create transaction B spending same inputs with higher fee, (4) Broadcast transaction B, (5) Miners prefer B (higher fee), A eventually dropped. Controversy: Enables double-spend of unconfirmed transactions. Merchants should never accept zero-confirmation from RBF transactions. Signaled in transaction sequence number.',
    },
    {
      id: 'tlm-mc-4',
      question:
        'Why do Bitcoin merchants typically require 6 confirmations for large payments?',
      options: [
        'Bitcoin protocol requires 6',
        'Each confirmation exponentially reduces reorg probability',
        '6 is lucky number',
        'It takes 6 blocks to validate',
      ],
      correctAnswer: 1,
      explanation:
        '6 confirmations ≈ 99.9%+ security. Each confirmation increases reorg difficulty exponentially. To reverse 1 confirmation: Attacker needs >50% hash power for short time. To reverse 6 confirmations: Needs >50% hash power for 6 blocks (expensive). Probability of successful 6-block reorg with 40% hash power: <0.1%. For small amounts: 1-2 confirmations OK. For large amounts ($10K+): 6+ confirmations. Exchanges typically require 3-6 confirmations for deposits. Trade-off: Security vs wait time (6 blocks = ~1 hour).',
    },
    {
      id: 'tlm-mc-5',
      question:
        'What happens to a transaction if it sits in the mempool for too long without being confirmed?',
      options: [
        'It is automatically confirmed eventually',
        'It stays in mempool forever',
        'Nodes eventually drop it from mempool (typically after ~2 weeks)',
        'It is returned to sender',
      ],
      correctAnswer: 2,
      explanation:
        'Mempool purge policy: Nodes eventually drop old unconfirmed transactions. Bitcoin Core default: 2 weeks. Configurable per node. Reasons: (1) Prevent mempool bloat, (2) Free resources, (3) User likely abandoned transaction. After purge: Transaction disappears, funds return to spendable state, user can create new transaction. No automatic confirmation or refund. If transaction important: Use RBF to bump fee before purge, or rebroadcast (nodes may reject duplicate), or create new transaction with higher fee.',
    },
  ];
