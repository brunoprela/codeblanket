/**
 * Crypto Topic Definition
 */

import { cryptoBlockchainFundamentalsModule } from '../modules/crypto-blockchain-fundamentals';

export const cryptoTopic = {
  id: 'crypto',
  title: 'Crypto',
  description:
    'Blockchain technology, cryptocurrency, smart contracts, and Web3 development',
  icon: '₿',
  modules: [cryptoBlockchainFundamentalsModule],
};
