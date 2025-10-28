export const transactionLifecycleMempoolDiscussion = [
  {
    id: 1,
    question:
      'During high network congestion, mempool fills with 100K+ pending transactions. Explain the fee market dynamics and how users can ensure transaction inclusion.',
    answer:
      'Congestion creates fee market auction. Miners prioritize highest fee transactions. User strategies: (1) Set high enough fee to be in top N transactions per block, (2) Use fee estimation APIs to predict required fee, (3) Use RBF to bump fee if stuck. EIP-1559 (Ethereum): Base fee rises automatically during congestion, users add priority tips. Example 2021 NFT mint: Gas prices hit 5000+ gwei, $500+ for simple transfer. Some transactions sat in mempool for days. Optimization: Batch transactions, use layer-2, or wait for off-peak hours.',
  },
  {
    id: 2,
    question:
      'Explain the security implications of Replace-by-Fee (RBF) for merchants accepting zero-confirmation transactions.',
    answer:
      'RBF enables fee bumping but also enables double-spend attacks. Attack: (1) Buy goods with low-fee transaction, (2) Merchant sees transaction in mempool, releases goods, (3) Attacker RBFs transaction to send funds to themselves with higher fee, (4) Miner includes replacement, merchant loses goods + payment. Defense: NEVER accept zero-confirmation for irreversible goods. Wait 1+ confirmations. Alternatively: Use child-pays-for-parent (CPFP) which is safer. RBF signaled in transaction—merchants can reject RBF transactions.',
  },
  {
    id: 3,
    question:
      "Compare Bitcoin's probabilistic finality vs Ethereum PoS's absolute finality. What are the trade-offs?",
    answer:
      "Bitcoin: 6 confirmations = 99.9%+ probability of permanence, but never 100%. Reorgs theoretically possible. Can still transact during network partition. Ethereum PoS: After 2 epochs finalized, reversal requires destroying 1/3+ of stake through slashing. Absolute finality. But if network partition prevents 2/3 consensus, finality halts. Trade-off: Bitcoin prioritizes liveness (always makes progress), Ethereum PoS prioritizes safety (no conflicting finality). Choice depends on use case: Payments (Bitcoin's approach OK), Settlement (Ethereum's finality better).",
  },
];
