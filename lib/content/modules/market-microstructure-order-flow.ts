import { Module } from '../../types';

// Import sections
import { marketMicrostructureFundamentals } from '@/lib/content/sections/market-microstructure-order-flow/market-microstructure-fundamentals';
import { orderBookDynamics } from '@/lib/content/sections/market-microstructure-order-flow/order-book-dynamics';
import { priceDiscoveryProcess } from '@/lib/content/sections/market-microstructure-order-flow/price-discovery-process';
import { bidAskSpreadDecomposition } from '@/lib/content/sections/market-microstructure-order-flow/bid-ask-spread-decomposition';
import { marketImpactSlippageModels } from '@/lib/content/sections/market-microstructure-order-flow/market-impact-slippage-models';
import { highFrequencyTrading } from '@/lib/content/sections/market-microstructure-order-flow/high-frequency-trading';
import { marketMakingLiquidityProvision } from '@/lib/content/sections/market-microstructure-order-flow/market-making-liquidity-provision';
import { orderFlowToxicity } from '@/lib/content/sections/market-microstructure-order-flow/order-flow-toxicity';
import { latencyColocation } from '@/lib/content/sections/market-microstructure-order-flow/latency-colocation';
import { darkPoolsAlternativeVenues } from '@/lib/content/sections/market-microstructure-order-flow/dark-pools-alternative-venues';
import { marketRegulations } from '@/lib/content/sections/market-microstructure-order-flow/market-regulations';
import { orderBookSimulator } from '@/lib/content/sections/market-microstructure-order-flow/order-book-simulator';

// Import quizzes
import { marketMicrostructureFundamentalsQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/market-microstructure-fundamentals';
import { orderBookDynamicsQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/order-book-dynamics';
import { priceDiscoveryProcessQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/price-discovery-process';
import { bidAskSpreadDecompositionQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/bid-ask-spread-decomposition';
import { marketImpactSlippageModelsQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/market-impact-slippage-models';
import { highFrequencyTradingQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/high-frequency-trading';
import { marketMakingLiquidityProvisionQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/market-making-liquidity-provision';
import { orderFlowToxicityQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/order-flow-toxicity';
import { latencyColocationQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/latency-colocation';
import { darkPoolsAlternativeVenuesQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/dark-pools-alternative-venues';
import { marketRegulationsQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/market-regulations';
import { orderBookSimulatorQuiz } from '@/lib/content/quizzes/market-microstructure-order-flow/order-book-simulator';

// Import multiple choice
import { marketMicrostructureFundamentalsMultipleChoice as marketMicrostructureFundamentalsMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/market-microstructure-fundamentals';
import { orderBookDynamicsMultipleChoice as orderBookDynamicsMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/order-book-dynamics';
import { priceDiscoveryProcessMultipleChoice as priceDiscoveryProcessMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/price-discovery-process';
import { bidAskSpreadDecompositionMultipleChoice as bidAskSpreadDecompositionMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/bid-ask-spread-decomposition';
import { marketImpactSlippageModelsMultipleChoice as marketImpactSlippageModelsMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/market-impact-slippage-models';
import { highFrequencyTradingMultipleChoice as highFrequencyTradingMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/high-frequency-trading';
import { marketMakingLiquidityProvisionMultipleChoice as marketMakingLiquidityProvisionMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/market-making-liquidity-provision';
import { orderFlowToxicityMultipleChoice as orderFlowToxicityMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/order-flow-toxicity';
import { latencyColocationMultipleChoice as latencyColocationMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/latency-colocation';
import { darkPoolsAlternativeVenuesMultipleChoice as darkPoolsAlternativeVenuesMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/dark-pools-alternative-venues';
import { marketRegulationsMultipleChoice as marketRegulationsMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/market-regulations';
import { orderBookSimulatorMultipleChoice as orderBookSimulatorMC } from '@/lib/content/multiple-choice/market-microstructure-order-flow/order-book-simulator';

// Transform functions
const transformQuiz = (quiz: any) => ({
  id: quiz.id,
  question: quiz.question,
  sampleAnswer: quiz.sampleAnswer,
  keyPoints: quiz.keyPoints,
});

const transformMC = (mc: any) => ({
  id: mc.id,
  question: mc.question,
  options: mc.options,
  correctAnswer: mc.correctAnswer,
  explanation: mc.explanation,
});

export const marketMicrostructureOrderFlowModule: Module = {
  id: 'market-microstructure-order-flow',
  title: 'Market Microstructure & Order Flow',
  description:
    'Master how markets work at the micro level: order books, HFT, market making, latency optimization, and building production matching engines',
  icon: '🔍',
  sections: [
    {
      id: marketMicrostructureFundamentals.id,
      title: marketMicrostructureFundamentals.title,
      content: marketMicrostructureFundamentals.content,
      quiz: marketMicrostructureFundamentalsQuiz.map(transformQuiz),
      multipleChoice: marketMicrostructureFundamentalsMC.map(transformMC),
    },
    {
      id: orderBookDynamics.id,
      title: orderBookDynamics.title,
      content: orderBookDynamics.content,
      quiz: orderBookDynamicsQuiz.map(transformQuiz),
      multipleChoice: orderBookDynamicsMC.map(transformMC),
    },
    {
      id: priceDiscoveryProcess.id,
      title: priceDiscoveryProcess.title,
      content: priceDiscoveryProcess.content,
      quiz: priceDiscoveryProcessQuiz.map(transformQuiz),
      multipleChoice: priceDiscoveryProcessMC.map(transformMC),
    },
    {
      id: bidAskSpreadDecomposition.id,
      title: bidAskSpreadDecomposition.title,
      content: bidAskSpreadDecomposition.content,
      quiz: bidAskSpreadDecompositionQuiz.map(transformQuiz),
      multipleChoice: bidAskSpreadDecompositionMC.map(transformMC),
    },
    {
      id: marketImpactSlippageModels.id,
      title: marketImpactSlippageModels.title,
      content: marketImpactSlippageModels.content,
      quiz: marketImpactSlippageModelsQuiz.map(transformQuiz),
      multipleChoice: marketImpactSlippageModelsMC.map(transformMC),
    },
    {
      id: highFrequencyTrading.id,
      title: highFrequencyTrading.title,
      content: highFrequencyTrading.content,
      quiz: highFrequencyTradingQuiz.map(transformQuiz),
      multipleChoice: highFrequencyTradingMC.map(transformMC),
    },
    {
      id: marketMakingLiquidityProvision.id,
      title: marketMakingLiquidityProvision.title,
      content: marketMakingLiquidityProvision.content,
      quiz: marketMakingLiquidityProvisionQuiz.map(transformQuiz),
      multipleChoice: marketMakingLiquidityProvisionMC.map(transformMC),
    },
    {
      id: orderFlowToxicity.id,
      title: orderFlowToxicity.title,
      content: orderFlowToxicity.content,
      quiz: orderFlowToxicityQuiz.map(transformQuiz),
      multipleChoice: orderFlowToxicityMC.map(transformMC),
    },
    {
      id: latencyColocation.id,
      title: latencyColocation.title,
      content: latencyColocation.content,
      quiz: latencyColocationQuiz.map(transformQuiz),
      multipleChoice: latencyColocationMC.map(transformMC),
    },
    {
      id: darkPoolsAlternativeVenues.id,
      title: darkPoolsAlternativeVenues.title,
      content: darkPoolsAlternativeVenues.content,
      quiz: darkPoolsAlternativeVenuesQuiz.map(transformQuiz),
      multipleChoice: darkPoolsAlternativeVenuesMC.map(transformMC),
    },
    {
      id: marketRegulations.id,
      title: marketRegulations.title,
      content: marketRegulations.content,
      quiz: marketRegulationsQuiz.map(transformQuiz),
      multipleChoice: marketRegulationsMC.map(transformMC),
    },
    {
      id: orderBookSimulator.id,
      title: orderBookSimulator.title,
      content: orderBookSimulator.content,
      quiz: orderBookSimulatorQuiz.map(transformQuiz),
      multipleChoice: orderBookSimulatorMC.map(transformMC),
    },
  ],
  quizzes: [
    ...marketMicrostructureFundamentalsQuiz.map(transformQuiz),
    ...orderBookDynamicsQuiz.map(transformQuiz),
    ...priceDiscoveryProcessQuiz.map(transformQuiz),
    ...bidAskSpreadDecompositionQuiz.map(transformQuiz),
    ...marketImpactSlippageModelsQuiz.map(transformQuiz),
    ...highFrequencyTradingQuiz.map(transformQuiz),
    ...marketMakingLiquidityProvisionQuiz.map(transformQuiz),
    ...orderFlowToxicityQuiz.map(transformQuiz),
    ...latencyColocationQuiz.map(transformQuiz),
    ...darkPoolsAlternativeVenuesQuiz.map(transformQuiz),
    ...marketRegulationsQuiz.map(transformQuiz),
    ...orderBookSimulatorQuiz.map(transformQuiz),
  ],
  multipleChoiceQuestions: [
    ...marketMicrostructureFundamentalsMC.map(transformMC),
    ...orderBookDynamicsMC.map(transformMC),
    ...priceDiscoveryProcessMC.map(transformMC),
    ...bidAskSpreadDecompositionMC.map(transformMC),
    ...marketImpactSlippageModelsMC.map(transformMC),
    ...highFrequencyTradingMC.map(transformMC),
    ...marketMakingLiquidityProvisionMC.map(transformMC),
    ...orderFlowToxicityMC.map(transformMC),
    ...latencyColocationMC.map(transformMC),
    ...darkPoolsAlternativeVenuesMC.map(transformMC),
    ...marketRegulationsMC.map(transformMC),
    ...orderBookSimulatorMC.map(transformMC),
  ],
};
