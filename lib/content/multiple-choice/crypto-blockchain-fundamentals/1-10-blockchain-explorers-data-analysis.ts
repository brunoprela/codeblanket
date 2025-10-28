import { MultipleChoiceQuestion } from '@/lib/types';

export const blockchainExplorersDataAnalysisMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'beda-mc-1',
      question: 'What is the primary purpose of a blockchain explorer?',
      options: [
        'To mine cryptocurrency',
        'To provide searchable interface for blockchain data',
        'To create new blocks',
        'To validate transactions',
      ],
      correctAnswer: 1,
      explanation:
        "Block explorers (Etherscan, Blockchain.com) index blockchain data into searchable databases. Users can: Query transactions by hash, View address balances and history, Explore blocks and their contents, Track transaction confirmations. Architecture: Full node → Indexer → Database → Web interface. Explorers don't mine or validate—they're read-only interfaces to blockchain data. Critical infrastructure for users checking transactions, developers debugging contracts, researchers analyzing on-chain activity.",
    },
    {
      id: 'beda-mc-2',
      question:
        'Why do block explorers use databases instead of querying the node directly for every request?',
      options: [
        "Nodes don't allow queries",
        'Database queries are much faster than node RPC calls',
        'Databases are more secure',
        "It's required by blockchain protocols",
      ],
      correctAnswer: 1,
      explanation:
        'Direct node queries too slow for user-facing applications. Node RPC: Linear scan through blocks/transactions. Database: Indexed queries in milliseconds. Example: Find all transactions for address. Node: Scan entire blockchain (hours). Database with indexes: <100ms. Explorers pre-index data: Address → transactions mapping, Transaction → block mapping, Contract → events mapping. Trade-off: Extra infrastructure (database) for massive performance gain. Real-time updates via block subscription + incremental indexing.',
    },
    {
      id: 'beda-mc-3',
      question: 'What is "address clustering" in blockchain analysis?',
      options: [
        'Grouping addresses by balance size',
        'Identifying multiple addresses controlled by same entity',
        'Geographic clustering of IP addresses',
        'Clustering mining pools',
      ],
      correctAnswer: 1,
      explanation:
        'Address clustering: Link pseudonymous addresses to real-world entities. Heuristics: (1) Common input ownership (addresses used as inputs in same transaction likely same owner), (2) Change address detection, (3) Address reuse patterns, (4) Co-spending patterns. Example: Exchange processes withdrawals from multiple addresses—analyst can cluster them as "Exchange X." Privacy implications: Blockchain pseudonymity breakable through analysis. Chainalysis and Elliptic use this for law enforcement. Defense: Avoid address reuse, use CoinJoin, privacy coins.',
    },
    {
      id: 'beda-mc-4',
      question:
        'How do block explorers handle blockchain reorganizations (reorgs)?',
      options: [
        'Ignore reorgs completely',
        'Track block depth and revert data if blocks orphaned',
        'Only show finalized blocks',
        'Shut down during reorgs',
      ],
      correctAnswer: 1,
      explanation:
        'Reorg handling: Store block depth (confirmations) with all data. When new block arrives: Check if extends longest chain or creates fork. If fork becomes longer: Revert database state to fork point, re-index canonical chain. Example: Block 100 initially has tx A, reorg happens, block 100 now has tx B—explorer must update. Implementation: Use database transactions for atomic updates, mark shallow blocks as "unconfirmed," typically show data as "final" after 6+ confirmations. This is why explorers sometimes show different data temporarily during reorgs.',
    },
    {
      id: 'beda-mc-5',
      question:
        'What data do blockchain explorers typically NOT have access to?',
      options: [
        'Transaction amounts',
        'Block timestamps',
        "Users' private keys and identity",
        'Smart contract code',
      ],
      correctAnswer: 2,
      explanation:
        'Explorers index public blockchain data: transactions, blocks, addresses, smart contracts, events. Cannot access: Private keys (never on blockchain), Real-world identities (blockchain is pseudonymous), Off-chain data (unless oracles), Encrypted transaction contents (privacy chains). What explorers CAN see: All public addresses, All transaction amounts, All transaction history, Smart contract code (verified source). Privacy implication: If you KYC at exchange and use that address publicly, anyone can see your full transaction history. This is why privacy matters and address rotation recommended.',
    },
  ];
