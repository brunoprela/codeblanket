import { cryptographicHashFunctions } from '../sections/crypto-blockchain-fundamentals/1-1-cryptographic-hash-functions';
import { publicKeyCryptographyDigitalSignatures } from '../sections/crypto-blockchain-fundamentals/1-2-public-key-cryptography-digital-signatures';
import { distributedConsensusFundamentals } from '../sections/crypto-blockchain-fundamentals/1-3-distributed-consensus-fundamentals';
import { bitcoinArchitectureDeepDive } from '../sections/crypto-blockchain-fundamentals/1-4-bitcoin-architecture-deep-dive';
import { proofOfWorkMining } from '../sections/crypto-blockchain-fundamentals/1-5-proof-of-work-mining';
import { proofOfStakeMechanisms } from '../sections/crypto-blockchain-fundamentals/1-6-proof-of-stake-mechanisms';
import { blockchainDataStructures } from '../sections/crypto-blockchain-fundamentals/1-7-blockchain-data-structures';
import { walletArchitectureKeyManagement } from '../sections/crypto-blockchain-fundamentals/1-8-wallet-architecture-key-management';
import { transactionLifecycleMempool } from '../sections/crypto-blockchain-fundamentals/1-9-transaction-lifecycle-mempool';
import { blockchainExplorersDataAnalysis } from '../sections/crypto-blockchain-fundamentals/1-10-blockchain-explorers-data-analysis';
import { nodeTypesNetworkArchitecture } from '../sections/crypto-blockchain-fundamentals/1-11-node-types-network-architecture';
import { buildingBlockchainFromScratch } from '../sections/crypto-blockchain-fundamentals/1-12-building-blockchain-from-scratch';

import { cryptographicHashFunctionsDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-1-cryptographic-hash-functions-discussion';
import { publicKeyCryptographyDigitalSignaturesDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-2-public-key-cryptography-digital-signatures-discussion';
import { distributedConsensusFundamentalsDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-3-distributed-consensus-fundamentals-discussion';
import { bitcoinArchitectureDeepDiveDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-4-bitcoin-architecture-deep-dive-discussion';
import { proofOfWorkMiningDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-5-proof-of-work-mining-discussion';
import { proofOfStakeMechanismsDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-6-proof-of-stake-mechanisms-discussion';
import { blockchainDataStructuresDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-7-blockchain-data-structures-discussion';
import { walletArchitectureKeyManagementDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-8-wallet-architecture-key-management-discussion';
import { transactionLifecycleMempoolDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-9-transaction-lifecycle-mempool-discussion';
import { blockchainExplorersDataAnalysisDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-10-blockchain-explorers-data-analysis-discussion';
import { nodeTypesNetworkArchitectureDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-11-node-types-network-architecture-discussion';
import { buildingBlockchainFromScratchDiscussion } from '../discussions/crypto-blockchain-fundamentals/1-12-building-blockchain-from-scratch-discussion';

import { cryptographicHashFunctionsMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-1-cryptographic-hash-functions';
import { publicKeyCryptographyDigitalSignaturesMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-2-public-key-cryptography-digital-signatures';
import { distributedConsensusFundamentalsMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-3-distributed-consensus-fundamentals';
import { bitcoinArchitectureDeepDiveMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-4-bitcoin-architecture-deep-dive';
import { proofOfWorkMiningMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-5-proof-of-work-mining';
import { proofOfStakeMechanismsMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-6-proof-of-stake-mechanisms';
import { blockchainDataStructuresMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-7-blockchain-data-structures';
import { walletArchitectureKeyManagementMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-8-wallet-architecture-key-management';
import { transactionLifecycleMempoolMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-9-transaction-lifecycle-mempool';
import { blockchainExplorersDataAnalysisMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-10-blockchain-explorers-data-analysis';
import { nodeTypesNetworkArchitectureMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-11-node-types-network-architecture';
import { buildingBlockchainFromScratchMultipleChoice } from '../multiple-choice/crypto-blockchain-fundamentals/1-12-building-blockchain-from-scratch';

export const cryptoBlockchainFundamentalsModule = {
  id: 'crypto-blockchain-fundamentals',
  title: 'Blockchain & Cryptography Fundamentals',
  description:
    'Master the core cryptographic primitives and blockchain architecture that power Bitcoin, Ethereum, and modern cryptocurrencies. Build production-grade blockchain systems from first principles.',
  icon: '⛓️',
  keyTakeaways: [
    'Understand cryptographic hash functions (SHA-256, collision resistance, Merkle trees)',
    'Master public key cryptography, ECDSA signatures, and key generation',
    'Solve distributed consensus (Byzantine Generals, CAP theorem, FLP impossibility)',
    'Deep dive into Bitcoin architecture: UTXO model, Script language, block structure',
    'Implement proof-of-work mining with difficulty adjustment and economic analysis',
    "Understand proof-of-stake: validator selection, slashing, Ethereum's Casper FFG",
    'Build efficient data structures: Merkle trees, Patricia tries, Bloom filters',
    'Design secure wallets: HD wallets (BIP32/39/44), multi-sig, hardware wallets',
    'Manage transaction lifecycle: creation, mempool, fee markets, RBF, finality',
    'Build blockchain explorers with indexers, APIs, and on-chain analytics',
    'Understand node types: full, archive, pruned, SPV nodes and P2P networks',
    'Build a complete blockchain from scratch with Python (blocks, PoW, wallets, networking)',
  ],
  sections: [
    {
      ...cryptographicHashFunctions,
      quiz: cryptographicHashFunctionsDiscussion,
      multipleChoice: cryptographicHashFunctionsMultipleChoice,
    },
    {
      ...publicKeyCryptographyDigitalSignatures,
      quiz: publicKeyCryptographyDigitalSignaturesDiscussion,
      multipleChoice: publicKeyCryptographyDigitalSignaturesMultipleChoice,
    },
    {
      ...distributedConsensusFundamentals,
      quiz: distributedConsensusFundamentalsDiscussion,
      multipleChoice: distributedConsensusFundamentalsMultipleChoice,
    },
    {
      ...bitcoinArchitectureDeepDive,
      quiz: bitcoinArchitectureDeepDiveDiscussion,
      multipleChoice: bitcoinArchitectureDeepDiveMultipleChoice,
    },
    {
      ...proofOfWorkMining,
      quiz: proofOfWorkMiningDiscussion,
      multipleChoice: proofOfWorkMiningMultipleChoice,
    },
    {
      ...proofOfStakeMechanisms,
      quiz: proofOfStakeMechanismsDiscussion,
      multipleChoice: proofOfStakeMechanismsMultipleChoice,
    },
    {
      ...blockchainDataStructures,
      quiz: blockchainDataStructuresDiscussion,
      multipleChoice: blockchainDataStructuresMultipleChoice,
    },
    {
      ...walletArchitectureKeyManagement,
      quiz: walletArchitectureKeyManagementDiscussion,
      multipleChoice: walletArchitectureKeyManagementMultipleChoice,
    },
    {
      ...transactionLifecycleMempool,
      quiz: transactionLifecycleMempoolDiscussion,
      multipleChoice: transactionLifecycleMempoolMultipleChoice,
    },
    {
      ...blockchainExplorersDataAnalysis,
      quiz: blockchainExplorersDataAnalysisDiscussion,
      multipleChoice: blockchainExplorersDataAnalysisMultipleChoice,
    },
    {
      ...nodeTypesNetworkArchitecture,
      quiz: nodeTypesNetworkArchitectureDiscussion,
      multipleChoice: nodeTypesNetworkArchitectureMultipleChoice,
    },
    {
      ...buildingBlockchainFromScratch,
      quiz: buildingBlockchainFromScratchDiscussion,
      multipleChoice: buildingBlockchainFromScratchMultipleChoice,
    },
  ],
};
