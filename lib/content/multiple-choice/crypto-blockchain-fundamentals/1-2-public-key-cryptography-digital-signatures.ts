import { MultipleChoiceQuestion } from '@/lib/types';

export const publicKeyCryptographyDigitalSignaturesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'pkc-mc-1',
      question:
        'In ECDSA signature generation, what catastrophic security breach occurs if the same nonce (k) is used to sign two different messages?',
      options: [
        'The signature becomes invalid and transactions are rejected',
        'The private key can be mathematically recovered by anyone observing both signatures',
        'The elliptic curve parameters become public knowledge',
        'The signature size doubles, making transactions more expensive',
      ],
      correctAnswer: 1,
      explanation:
        "Nonce reuse in ECDSA reveals the private key through simple algebra. Given two signatures (r₁,s₁) and (r₂,s₂) with same r (indicating same k), an attacker can: (1) Calculate k = (hash₁ - hash₂)/(s₁ - s₂), (2) Calculate private_key = (s×k - hash)/r. Real example: Sony PlayStation 3 (2010) was hacked because firmware signing used static k. Android Bitcoin wallets (2013) lost $5.7M due to weak RNG producing duplicate k values. This is why Bitcoin uses RFC 6979 deterministic k generation: k = HMAC(private_key, message) ensuring unique k per message without needing entropy. RFC 6979 prevents random number failures while maintaining security. Never reuse nonces - it's cryptographic suicide!",
    },
    {
      id: 'pkc-mc-2',
      question:
        'What is the primary advantage of Ed25519 over secp256k1 (used in Bitcoin) that makes it attractive for high-throughput blockchains like Solana?',
      options: [
        'Ed25519 is quantum-resistant while secp256k1 is not',
        'Ed25519 produces smaller signatures (32 bytes vs 64 bytes)',
        'Ed25519 is 2-4x faster for signature verification and uses deterministic nonces',
        'Ed25519 is mathematically proven secure while secp256k1 relies on unproven assumptions',
      ],
      correctAnswer: 2,
      explanation:
        "Ed25519's key advantage is performance: signature verification is ~2.3x faster than secp256k1 ECDSA, and signing is ~4x faster. This is critical for blockchains targeting high TPS (Solana: 65,000 TPS). Ed25519 uses deterministic nonces (no random k needed), eliminating nonce reuse vulnerabilities. Both are quantum-vulnerable (Shor's algorithm), so answer A is wrong. Ed25519 signatures are 64 bytes, same as compact secp256k1 (answer B wrong). Both rely on discrete logarithm hardness (answer D wrong). Speed breakdown: secp256k1 verification ~0.3ms, Ed25519 ~0.13ms. At 100,000 TPS: secp256k1 = 30 seconds verification time per batch, Ed25519 = 13 seconds. That 17-second difference is why Solana chose Ed25519. Additional benefits: simpler implementation (fewer bugs), no need for secure random number generator during signing.",
    },
    {
      id: 'pkc-mc-3',
      question:
        'Why does Bitcoin derive addresses from the hash of the public key (P2PKH) rather than using the public key directly as the address?',
      options: [
        'To make addresses shorter and more user-friendly',
        'To provide post-quantum security - even if ECDSA is broken, the hash provides protection until funds are spent',
        'To enable multiple signatures to share the same address',
        'To reduce blockchain storage by not storing full public keys',
      ],
      correctAnswer: 1,
      explanation:
        "P2PKH (Pay to Public Key Hash) provides additional security layer against quantum computers. The public key is hidden until you spend funds - only the hash is visible on blockchain. Even if quantum computer can derive private key from public key (Shor's algorithm), attacker first needs to reverse the hash (preimage attack on SHA-256 + RIPEMD-160) which remains quantum-resistant. This gives temporary protection: (1) Before spending: Quantum attacker cannot find public key from address hash, (2) After spending: Public key exposed in transaction, vulnerable to quantum attack. Historical note: Early Bitcoin used P2PK (public key directly) - Satoshi's 1M BTC uses P2PK and is quantum-vulnerable. While shorter addresses (answer A) is true (25 bytes vs 65 bytes uncompressed public key), security is the primary reason. Answers C and D are incorrect - P2SH/P2WSH handle multisig, and public keys must still be revealed when spending.",
    },
    {
      id: 'pkc-mc-4',
      question:
        'In elliptic curve cryptography, the "discrete logarithm problem" refers to which computational challenge?',
      options: [
        'Computing log(x) for very large numbers x',
        'Given points P and Q = k×P, finding the scalar k',
        'Finding the inverse of an elliptic curve point',
        'Computing the logarithm of the curve order modulo a prime',
      ],
      correctAnswer: 1,
      explanation:
        'The elliptic curve discrete logarithm problem (ECDLP): given generator point P and public key Q = k×P, find private key k. The "×" here means scalar multiplication on elliptic curve (P+P+...+P k times), not regular multiplication. Forward direction (k → Q) is easy: ~256 point additions for 256-bit k using double-and-add algorithm. Reverse direction (Q → k) is hard: best classical algorithm (Pollard\'s Rho) needs ~√n operations = 2^128 for 256-bit curve. This is why Bitcoin is secure: deriving public key from private key takes milliseconds, but reverse would take 10^31 years. Quantum threat: Shor\'s algorithm solves ECDLP in polynomial time O((log n)³) ≈ 2^36 operations for 256-bit curve, reducing security to hours instead of trillions of years. Real numbers: Classical: 2^128 ops = 10^31 years. Quantum: 2^36 ops = ~1 hour. This asymmetry enables public key cryptography.',
    },
    {
      id: 'pkc-mc-5',
      question:
        'What is the fundamental difference between how Bitcoin and Ethereum derive addresses from public keys?',
      options: [
        'Bitcoin uses Base58 encoding while Ethereum uses hexadecimal',
        'Bitcoin uses SHA-256+RIPEMD-160 hash, Ethereum uses Keccak-256 hash',
        'Bitcoin derives from compressed public keys, Ethereum from uncompressed',
        'Bitcoin includes a checksum in the address, Ethereum does not',
      ],
      correctAnswer: 1,
      explanation:
        'Core difference is hash function: Bitcoin: SHA-256(public_key) then RIPEMD-160(result) = 20-byte hash. Ethereum: Keccak-256(public_key) then take last 20 bytes. Why different? (1) Bitcoin chose double hashing (SHA-256+RIPEMD-160) for additional security margin and smaller addresses. (2) Ethereum chose Keccak-256 (pre-standard SHA-3) for simplicity and speed. Note: Ethereum uses Keccak-256, NOT NIST SHA3-256 (they differ slightly). Additional differences: Bitcoin addresses are Base58Check encoded (e.g., 1A1zP1...) with checksum. Ethereum addresses are hex (0x...) without built-in checksum (though EIP-55 adds optional checksum via mixed case). Both use secp256k1 curve for public keys. Answer A is encoding (consequence), B is fundamental difference. Answer C is wrong - both support compressed keys. Answer D - Bitcoin has checksum in address itself, Ethereum added EIP-55 checksumming later.',
    },
  ];
