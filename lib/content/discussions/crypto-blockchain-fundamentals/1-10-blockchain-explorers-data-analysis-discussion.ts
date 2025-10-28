export const blockchainExplorersDataAnalysisDiscussion = [
  {
    id: 1,
    question:
      'Design the architecture for a block explorer that handles 10M requests/day.',
    answer:
      'Architecture: (1) Multiple full nodes for redundancy, (2) Indexer service (Kafka/RabbitMQ for queue), (3) PostgreSQL with partitioning (by block height), (4) Redis cache (hot data), (5) CDN for static assets, (6) Load balancer, (7) Rate limiting. Indexer: Process blocks sequentially, handle reorgs by storing depth, batch writes to DB. API: Cache common queries (address balance, recent blocks), paginate results. Scaling: Read replicas for DB, horizontal scaling of API servers. Cost: ~$10K/month for infrastructure.',
  },
  {
    id: 2,
    question:
      "Explain how to track 'flow of funds' from one address to another on-chain.",
    answer:
      'Graph traversal problem. Build transaction graph: addresses are nodes, transactions are edges. BFS/DFS from source to destination. Challenges: (1) Graph is huge (millions of addresses), (2) Multiple paths possible, (3) Privacy (mixing, exchanges break trail). Tools: Neo4j graph database, custom indexer. Example: Track ransomware payment—follow transactions until exchange deposit (trail goes cold). Commercial tools: Chainalysis, Elliptic. Privacy coins (Monero, Zcash) make this impossible.',
  },
  {
    id: 3,
    question:
      'What data privacy concerns exist with public blockchain explorers?',
    answer:
      "Blockchains are pseudonymous, not anonymous. Block explorers enable: (1) Address balance surveillance, (2) Transaction history analysis, (3) Address clustering (linking related addresses), (4) Identity linking (if address KYC'd at exchange). Concerns: Financial privacy, targeted attacks (know wealthy addresses), censorship (governments track dissidents). Mitigations: Use fresh addresses, CoinJoin mixing, privacy coins, Lightning Network (off-chain). Trade-off: Transparency for auditability vs privacy for users.",
  },
];
