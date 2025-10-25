import { designingOrderManagementSystems } from '../sections/system-design-trading/designing-order-management-systems';
import { designingMarketDataSystems } from '../sections/system-design-trading/designing-market-data-systems';
import { designingBacktestingEngines } from '../sections/system-design-trading/designing-backtesting-engines';
import { designingRiskSystems } from '../sections/system-design-trading/designing-risk-systems';
import { highFrequencyTradingArchitecture } from '../sections/system-design-trading/high-frequency-trading-architecture';
import { distributedTradingSystems } from '../sections/system-design-trading/distributed-trading-systems';
import { regulatoryComplianceSystems } from '../sections/system-design-trading/regulatory-compliance-systems';
import { mlModelServingTrading } from '../sections/system-design-trading/ml-model-serving-trading';

import { designingOrderManagementSystemsQuiz } from '../quizzes/system-design-trading/designing-order-management-systems';
import { designingMarketDataSystemsQuiz } from '../quizzes/system-design-trading/designing-market-data-systems';
import { designingBacktestingEnginesQuiz } from '../quizzes/system-design-trading/designing-backtesting-engines';
import { designingRiskSystemsQuiz } from '../quizzes/system-design-trading/designing-risk-systems';
import { highFrequencyTradingArchitectureQuiz } from '../quizzes/system-design-trading/high-frequency-trading-architecture';
import { distributedTradingSystemsQuiz } from '../quizzes/system-design-trading/distributed-trading-systems';
import { regulatoryComplianceSystemsQuiz } from '../quizzes/system-design-trading/regulatory-compliance-systems';
import { mlModelServingTradingQuiz } from '../quizzes/system-design-trading/ml-model-serving-trading';

import { designingOrderManagementSystemsMultipleChoice } from '../multiple-choice/system-design-trading/designing-order-management-systems';
import { designingMarketDataSystemsMultipleChoice } from '../multiple-choice/system-design-trading/designing-market-data-systems';
import { designingBacktestingEnginesMultipleChoice } from '../multiple-choice/system-design-trading/designing-backtesting-engines';
import { designingRiskSystemsMultipleChoice } from '../multiple-choice/system-design-trading/designing-risk-systems';
import { highFrequencyTradingArchitectureMultipleChoice } from '../multiple-choice/system-design-trading/high-frequency-trading-architecture';
import { distributedTradingSystemsMultipleChoice } from '../multiple-choice/system-design-trading/distributed-trading-systems';
import { regulatoryComplianceSystemsMultipleChoice } from '../multiple-choice/system-design-trading/regulatory-compliance-systems';
import { mlModelServingTradingMultipleChoice } from '../multiple-choice/system-design-trading/ml-model-serving-trading';

export const systemDesignTradingModule = {
  id: 'system-design-trading',
  title: 'System Design for Trading Systems',
  description:
    'Design production trading systems including order management, market data, backtesting, risk management, HFT architecture, distributed systems, compliance, and ML model serving. Learn to build systems handling millions of orders with microsecond latency requirements.',
  icon: '💹',
  keyTakeaways: [
    'Design order management systems with <1ms latency handling 100K+ orders/sec',
    'Build market data systems ingesting and normalizing 1M+ ticks/sec with sub-millisecond processing',
    'Create event-driven backtesting engines with vectorized operations for 100x speedup',
    'Implement real-time risk systems computing P&L, VaR, and Greeks across 10K+ positions',
    'Architect high-frequency trading systems with sub-microsecond latency using FPGA and kernel bypass',
    'Design distributed trading systems with multi-region deployment handling CAP theorem trade-offs',
    'Build regulatory compliance systems with immutable audit trails and trade surveillance',
    'Deploy ML models for trading with <1ms inference using ONNX Runtime and Redis feature stores',
  ],
  sections: [
    {
      ...designingOrderManagementSystems,
      quiz: designingOrderManagementSystemsQuiz,
      multipleChoice: designingOrderManagementSystemsMultipleChoice,
    },
    {
      ...designingMarketDataSystems,
      quiz: designingMarketDataSystemsQuiz,
      multipleChoice: designingMarketDataSystemsMultipleChoice,
    },
    {
      ...designingBacktestingEngines,
      quiz: designingBacktestingEnginesQuiz,
      multipleChoice: designingBacktestingEnginesMultipleChoice,
    },
    {
      ...designingRiskSystems,
      quiz: designingRiskSystemsQuiz,
      multipleChoice: designingRiskSystemsMultipleChoice,
    },
    {
      ...highFrequencyTradingArchitecture,
      quiz: highFrequencyTradingArchitectureQuiz,
      multipleChoice: highFrequencyTradingArchitectureMultipleChoice,
    },
    {
      ...distributedTradingSystems,
      quiz: distributedTradingSystemsQuiz,
      multipleChoice: distributedTradingSystemsMultipleChoice,
    },
    {
      ...regulatoryComplianceSystems,
      quiz: regulatoryComplianceSystemsQuiz,
      multipleChoice: regulatoryComplianceSystemsMultipleChoice,
    },
    {
      ...mlModelServingTrading,
      quiz: mlModelServingTradingQuiz,
      multipleChoice: mlModelServingTradingMultipleChoice,
    },
  ],
};
