/**
 * Financial Markets & Instruments Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { equityMarketsDeepDive } from '../sections/financial-markets-instruments/equity-markets-deep-dive';
import { fixedIncomeMarkets } from '../sections/financial-markets-instruments/fixed-income-markets';
import { derivativesOverview } from '../sections/financial-markets-instruments/derivatives-overview';
import { foreignExchangeMarkets } from '../sections/financial-markets-instruments/foreign-exchange-markets';
import { commoditiesMarkets } from '../sections/financial-markets-instruments/commodities-markets';
import { cryptocurrencyMarkets } from '../sections/financial-markets-instruments/cryptocurrency-markets';
import { etfsMutualFunds } from '../sections/financial-markets-instruments/etfs-mutual-funds';
import { alternativeInvestments } from '../sections/financial-markets-instruments/alternative-investments';
import { marketParticipants } from '../sections/financial-markets-instruments/market-participants';
import { tradingVenuesExchanges } from '../sections/financial-markets-instruments/trading-venues-exchanges';
import { orderTypesExecution } from '../sections/financial-markets-instruments/order-types-execution';
import { marketDataPriceDiscovery } from '../sections/financial-markets-instruments/market-data-price-discovery';
import { liquidityMarketImpact } from '../sections/financial-markets-instruments/liquidity-market-impact';
import { moduleProjectMarketDataDashboard } from '../sections/financial-markets-instruments/module-project-market-data-dashboard';

// Import quizzes
import { equityMarketsDeepDiveQuiz } from '../quizzes/financial-markets-instruments/equity-markets-deep-dive';
import { fixedIncomeMarketsQuiz } from '../quizzes/financial-markets-instruments/fixed-income-markets';
import { derivativesOverviewQuiz } from '../quizzes/financial-markets-instruments/derivatives-overview';
import { foreignExchangeMarketsQuiz } from '../quizzes/financial-markets-instruments/foreign-exchange-markets';
import { commoditiesMarketsQuiz } from '../quizzes/financial-markets-instruments/commodities-markets';
import { cryptocurrencyMarketsQuiz } from '../quizzes/financial-markets-instruments/cryptocurrency-markets';
import { etfsMutualFundsQuiz } from '../quizzes/financial-markets-instruments/etfs-mutual-funds';
import { alternativeInvestmentsQuiz } from '../quizzes/financial-markets-instruments/alternative-investments';
import { marketParticipantsQuiz } from '../quizzes/financial-markets-instruments/market-participants';
import { tradingVenuesExchangesQuiz } from '../quizzes/financial-markets-instruments/trading-venues-exchanges';
import { orderTypesExecutionQuiz } from '../quizzes/financial-markets-instruments/order-types-execution';
import { marketDataPriceDiscoveryQuiz } from '../quizzes/financial-markets-instruments/market-data-price-discovery';
import { liquidityMarketImpactQuiz } from '../quizzes/financial-markets-instruments/liquidity-market-impact';
import { moduleProjectMarketDataDashboardQuiz } from '../quizzes/financial-markets-instruments/module-project-market-data-dashboard';

// Import multiple choice
import { equityMarketsDeepDiveMultipleChoice } from '../multiple-choice/financial-markets-instruments/equity-markets-deep-dive';
import { fixedIncomeMarketsMultipleChoice } from '../multiple-choice/financial-markets-instruments/fixed-income-markets';
import { derivativesOverviewMultipleChoice } from '../multiple-choice/financial-markets-instruments/derivatives-overview';
import { foreignExchangeMarketsMultipleChoice } from '../multiple-choice/financial-markets-instruments/foreign-exchange-markets';
import { commoditiesMarketsMultipleChoice } from '../multiple-choice/financial-markets-instruments/commodities-markets';
import { cryptocurrencyMarketsMultipleChoice } from '../multiple-choice/financial-markets-instruments/cryptocurrency-markets';
import { etfsMutualFundsMultipleChoice } from '../multiple-choice/financial-markets-instruments/etfs-mutual-funds';
import { alternativeInvestmentsMultipleChoice } from '../multiple-choice/financial-markets-instruments/alternative-investments';
import { marketParticipantsMultipleChoice } from '../multiple-choice/financial-markets-instruments/market-participants';
import { tradingVenuesExchangesMultipleChoice } from '../multiple-choice/financial-markets-instruments/trading-venues-exchanges';
import { orderTypesExecutionMultipleChoice } from '../multiple-choice/financial-markets-instruments/order-types-execution';
import { marketDataPriceDiscoveryMultipleChoice } from '../multiple-choice/financial-markets-instruments/market-data-price-discovery';
import { liquidityMarketImpactMultipleChoice } from '../multiple-choice/financial-markets-instruments/liquidity-market-impact';
import { moduleProjectMarketDataDashboardMultipleChoice } from '../multiple-choice/financial-markets-instruments/module-project-market-data-dashboard';

export const financialMarketsInstrumentsModule: Module = {
  id: 'financial-markets-instruments',
  title: 'Financial Markets & Instruments',
  description:
    'Master financial markets, trading instruments, market microstructure, and build a complete market data dashboard',
  icon: '🏛️',
  estimatedHours: 50,
  prerequisites: [],
  learningObjectives: [
    'Understand equity, fixed income, derivatives, FX, and commodities markets',
    'Master market microstructure, liquidity, and price discovery mechanisms',
    'Learn order types, execution algorithms, and smart order routing',
    'Build production-ready market data systems and trading dashboards',
    'Analyze real-world trading scenarios and risk management strategies',
  ],
  sections: [
    {
      ...equityMarketsDeepDive,
      id: equityMarketsDeepDive.slug,
      quiz: equityMarketsDeepDiveQuiz,
      multipleChoice: equityMarketsDeepDiveMultipleChoice,
    },
    {
      ...fixedIncomeMarkets,
      id: fixedIncomeMarkets.slug,
      quiz: fixedIncomeMarketsQuiz,
      multipleChoice: fixedIncomeMarketsMultipleChoice,
    },
    {
      ...derivativesOverview,
      id: derivativesOverview.slug,
      quiz: derivativesOverviewQuiz,
      multipleChoice: derivativesOverviewMultipleChoice,
    },
    {
      ...foreignExchangeMarkets,
      id: foreignExchangeMarkets.slug,
      quiz: foreignExchangeMarketsQuiz,
      multipleChoice: foreignExchangeMarketsMultipleChoice,
    },
    {
      ...commoditiesMarkets,
      id: commoditiesMarkets.slug,
      quiz: commoditiesMarketsQuiz,
      multipleChoice: commoditiesMarketsMultipleChoice,
    },
    {
      ...cryptocurrencyMarkets,
      id: cryptocurrencyMarkets.slug,
      quiz: cryptocurrencyMarketsQuiz,
      multipleChoice: cryptocurrencyMarketsMultipleChoice,
    },
    {
      ...etfsMutualFunds,
      id: etfsMutualFunds.slug,
      quiz: etfsMutualFundsQuiz,
      multipleChoice: etfsMutualFundsMultipleChoice,
    },
    {
      ...alternativeInvestments,
      id: alternativeInvestments.slug,
      quiz: alternativeInvestmentsQuiz,
      multipleChoice: alternativeInvestmentsMultipleChoice,
    },
    {
      ...marketParticipants,
      id: marketParticipants.slug,
      quiz: marketParticipantsQuiz,
      multipleChoice: marketParticipantsMultipleChoice,
    },
    {
      ...tradingVenuesExchanges,
      id: tradingVenuesExchanges.slug,
      quiz: tradingVenuesExchangesQuiz,
      multipleChoice: tradingVenuesExchangesMultipleChoice,
    },
    {
      ...orderTypesExecution,
      id: orderTypesExecution.slug,
      quiz: orderTypesExecutionQuiz,
      multipleChoice: orderTypesExecutionMultipleChoice,
    },
    {
      ...marketDataPriceDiscovery,
      id: marketDataPriceDiscovery.slug,
      quiz: marketDataPriceDiscoveryQuiz,
      multipleChoice: marketDataPriceDiscoveryMultipleChoice,
    },
    {
      ...liquidityMarketImpact,
      id: liquidityMarketImpact.slug,
      quiz: liquidityMarketImpactQuiz,
      multipleChoice: liquidityMarketImpactMultipleChoice,
    },
    {
      ...moduleProjectMarketDataDashboard,
      id: moduleProjectMarketDataDashboard.slug,
      quiz: moduleProjectMarketDataDashboardQuiz,
      multipleChoice: moduleProjectMarketDataDashboardMultipleChoice,
    },
  ],
};
