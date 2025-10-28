import { MultipleChoiceQuestion } from '@/lib/types';

export const walletArchitectureKeyManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'wakm-mc-1',
      question: 'What does "HD" mean in HD wallets?',
      options: [
        'High Definition',
        'Hierarchical Deterministic',
        'Hardware Device',
        'Hash Derived',
      ],
      correctAnswer: 1,
      explanation:
        'HD = Hierarchical Deterministic. Hierarchical: Tree structure of keys (master→child→grandchild). Deterministic: Keys derived from seed using algorithm, not random. BIP32 standard: Seed → Master private key → Child keys via HMAC-SHA512. Benefit: Backup single seed phrase (12-24 words), derive infinite keys. vs Non-HD: Generate random keys, must backup each one individually. Modern wallets (MetaMask, Ledger) all use HD wallets.',
    },
    {
      id: 'wakm-mc-2',
      question:
        'In a 2-of-3 multi-signature wallet, how many signatures are needed to spend funds?',
      options: [
        '2 signatures out of 3 possible signers',
        '3 signatures required',
        '2 or 3 signatures',
        'Depends on transaction amount',
      ],
      correctAnswer: 0,
      explanation:
        '2-of-3 multi-sig: Need ANY 2 signatures from 3 authorized keys. Use case: Personal security—keys on laptop, phone, hardware wallet. Lose one device (or one compromised), still access funds with other two. Cannot spend with only one key. More secure than single key (no single point of failure) but less convenient (need coordinate 2 devices). Common configurations: 2-of-3 (personal), 3-of-5 (corporate), 5-of-7 (exchanges).',
    },
    {
      id: 'wakm-mc-3',
      question: 'What is a "seed phrase" in cryptocurrency wallets?',
      options: [
        'A password to encrypt the wallet',
        '12-24 words that encode the master seed for an HD wallet',
        'The first transaction in a wallet',
        'A security question',
      ],
      correctAnswer: 1,
      explanation:
        'Seed phrase (BIP39): 12-24 words from standard wordlist, encodes 128-256 bit random seed. Process: Random entropy → Checksum → Map to words. Example: "abandon abandon abandon ... about" (valid test seed). Seed → Master private key → All wallet keys. CRITICAL: Anyone with seed phrase can recreate entire wallet and steal all funds. Write on paper, store securely. NEVER digital storage, NEVER photos, NEVER email. Seed phrase IS the wallet—not just password, but the keys themselves.',
    },
    {
      id: 'wakm-mc-4',
      question:
        'What is the main security advantage of hardware wallets over software wallets?',
      options: [
        'Hardware wallets are faster',
        'Private keys never leave the device, signing happens internally',
        'Hardware wallets have more storage',
        "Hardware wallets don't need backups",
      ],
      correctAnswer: 1,
      explanation:
        "Hardware wallet security: Private keys generated and stored on device, NEVER exposed to computer. Signing process: (1) Computer sends transaction to hardware wallet, (2) Hardware wallet signs internally, (3) Returns signed transaction (not private key). Even if computer has malware, attacker can't steal private keys. Trade-off: Less convenient (physical device needed) but vastly more secure. Software wallet: Keys on computer, vulnerable to malware. Hardware wallets still need backups (seed phrase on paper).",
    },
    {
      id: 'wakm-mc-5',
      question:
        'Why is it dangerous to generate a Bitcoin private key from a "brain wallet" (password/phrase)?',
      options: [
        'Brain wallets are illegal',
        'Human-chosen phrases have low entropy and are easily brute-forced',
        'Brain wallets only work with Ethereum',
        'Brain wallets require special hardware',
      ],
      correctAnswer: 1,
      explanation:
        'Brain wallets let users create wallet from memorable phrase. Sounds convenient, but EXTREMELY dangerous. Problem: Humans choose predictable phrases. Research shows: Common phrases like "correct horse battery staple" were robbed within days. Attackers pre-compute addresses for millions of common phrases, monitor blockchain, instantly steal any funds. True 128-bit entropy: 2^128 possible keys. Human phrase: Maybe 2^40 effective entropy (easily brute-forced). Solution: Use cryptographically random seed generation (os.urandom), derive seed phrase from randomness, not vice versa. NEVER trust human-generated randomness for cryptography.',
    },
  ];
