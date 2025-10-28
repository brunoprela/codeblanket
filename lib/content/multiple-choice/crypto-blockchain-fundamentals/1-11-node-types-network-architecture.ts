import { MultipleChoiceQuestion } from '@/lib/types';

export const nodeTypesNetworkArchitectureMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'ntna-mc-1',
      question:
        'What is the main difference between a full node and an archive node?',
      options: [
        "Full nodes validate blocks, archive nodes don't",
        'Archive nodes store all historical states, full nodes only current state',
        'Full nodes are faster',
        'Archive nodes require special hardware',
      ],
      correctAnswer: 1,
      explanation:
        "Full node: Validates all transactions, stores current UTXO/state, prunes old data. Archive node: Full node + ALL historical states (every account balance at every block). Storage: Bitcoin full node ~500GB (pruned ~5GB), archive node ~500GB+. Ethereum full node ~500GB, archive node ~12TB+. Use cases: Full node for validation/wallets. Archive node for blockchain explorers, historical queries, dApp development. Trade-off: Archive nodes 10-20× storage for historical data access. Most users don't need archive node—full node sufficient.",
    },
    {
      id: 'ntna-mc-2',
      question: "How does Bitcoin's P2P network achieve message propagation?",
      options: [
        'Central server broadcasts to all nodes',
        'Gossip protocol—each node forwards to peers',
        'Direct connections between all nodes',
        'Blockchain is mailed on USB drives',
      ],
      correctAnswer: 1,
      explanation:
        "Gossip protocol (flooding): Node receives message → validates → forwards to all peers. Each peer repeats process. Result: Message reaches entire network in seconds. Example: Miner finds block → broadcasts to 8 peers → each broadcasts to their 8 peers → exponential spread. ~5 seconds to reach 99% of network. Advantages: (1) No single point of failure, (2) Censorship resistant, (3) Works with changing topology. Disadvantages: (1) Redundant messages (bandwidth), (2) No guarantees (network partitions possible). Optimizations: Don't forward to sender, track seen messages, compact blocks.",
    },
    {
      id: 'ntna-mc-3',
      question: 'What is an "eclipse attack" on a blockchain node?',
      options: [
        'Attacking during solar eclipse',
        'Isolating node from honest network by controlling all its connections',
        "Stealing node's private keys",
        'DDoS attack on node',
      ],
      correctAnswer: 1,
      explanation:
        "Eclipse attack: Attacker surrounds target node with malicious nodes, cutting it off from honest network. Process: (1) Sybil attack—create many nodes, (2) Monopolize target's connections, (3) Feed target fake blockchain. Consequences: (1) Target accepts invalid transactions, (2) Double-spend possible, (3) Mining on wrong chain. Defense: (1) Diverse peer selection (don't connect to all nodes in same /16 subnet), (2) Long-lived connections, (3) Prefer outbound connections, (4) Node restart vulnerable—attacker can flood during connection phase. Real-world: Difficult against well-connected nodes, easier against new/restarting nodes.",
    },
    {
      id: 'ntna-mc-4',
      question:
        'Why do light clients (SPV) have weaker security than full nodes?',
      options: [
        "Light clients can't make transactions",
        "Light clients don't validate all transactions, trust miners' longest chain",
        'Light clients are slower',
        'Light clients require internet connection',
      ],
      correctAnswer: 1,
      explanation:
        'SPV weakness: Downloads headers only (~80 bytes vs ~1MB per block), validates proof-of-work but NOT transactions. Trusts miners: If miners create invalid block with invalid transactions but valid PoW, SPV client accepts it. Requires 51% attack (expensive) but still weaker than full node validation. Also: (1) Privacy—SPV client reveals addresses to full nodes when requesting Merkle proofs, (2) Eclipse attack vulnerable—fake full nodes can feed wrong chain. Full node: Validates EVERY transaction, detects invalid blocks even with valid PoW. Trade-off: SPV = convenience (64MB), Full node = security (500GB).',
    },
    {
      id: 'ntna-mc-5',
      question: 'What resources are required to run a Bitcoin full node?',
      options: [
        'Minimal—any computer works',
        '500GB+ storage, 10GB+ RAM, good internet',
        'Supercomputer with specialized hardware',
        'Just a smartphone',
      ],
      correctAnswer: 1,
      explanation:
        "Bitcoin full node requirements (2024): Storage: 500GB+ (growing ~50GB/year), or 5GB pruned mode. RAM: 4GB minimum, 8-16GB recommended. Bandwidth: 200GB+/month (upload heavy). CPU: Modern multi-core (validation CPU-intensive). Initial sync: ~1-2 days. Ethereum more demanding: 500GB+ storage (full), 12TB+ (archive), 16GB+ RAM. Why run node: (1) Trustless validation, (2) Privacy (don't rely on third parties), (3) Support network, (4) Required for mining. Most users don't run nodes—use SPV wallets or trusted services. Trade-off: Resource cost vs security/decentralization.",
    },
  ];
