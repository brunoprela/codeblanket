/**
 * Module: Risk Management & Portfolio Systems
 * Module 15 of Finance Curriculum
 */

import { Module, DiscussionItem, QuizQuestion } from '../../types';

// Section imports
import { riskManagementFundamentals } from '../sections/risk-management-portfolio-systems/risk-management-fundamentals';
import { valueAtRiskMethods } from '../sections/risk-management-portfolio-systems/value-at-risk-methods';
import { conditionalValueAtRisk } from '../sections/risk-management-portfolio-systems/conditional-value-at-risk';
import { stressTestingScenarioAnalysis } from '../sections/risk-management-portfolio-systems/stress-testing-scenario-analysis';
import { marketRiskManagement } from '../sections/risk-management-portfolio-systems/market-risk-management';
import { creditRiskManagement } from '../sections/risk-management-portfolio-systems/credit-risk-management';
import { operationalRisk } from '../sections/risk-management-portfolio-systems/operational-risk';
import { liquidityRisk } from '../sections/risk-management-portfolio-systems/liquidity-risk';
import { riskAttributionAnalysis } from '../sections/risk-management-portfolio-systems/risk-attribution-analysis';
import { riskBudgeting } from '../sections/risk-management-portfolio-systems/risk-budgeting';
import { marginCollateralManagement } from '../sections/risk-management-portfolio-systems/margin-collateral-management';
import { positionLimitsRiskLimits } from '../sections/risk-management-portfolio-systems/position-limits-risk-limits';
import { realTimeRiskMonitoring } from '../sections/risk-management-portfolio-systems/real-time-risk-monitoring';
import { riskReportingDashboards } from '../sections/risk-management-portfolio-systems/risk-reporting-dashboards';
import { blackrockAladdinArchitecture } from '../sections/risk-management-portfolio-systems/blackrock-aladdin-architecture';
import { riskManagementPlatformProject } from '../sections/risk-management-portfolio-systems/risk-management-platform-project';

// Discussion imports
import finM15S1Discussion from '../discussions/finance/fin-m15-s1-discussion';
import finM15S2Discussion from '../discussions/finance/fin-m15-s2-discussion';
import finM15S3Discussion from '../discussions/finance/fin-m15-s3-discussion';
import finM15S4Discussion from '../discussions/finance/fin-m15-s4-discussion';
import finM15S5Discussion from '../discussions/finance/fin-m15-s5-discussion';
import finM15S6Discussion from '../discussions/finance/fin-m15-s6-discussion';
import finM15S7Discussion from '../discussions/finance/fin-m15-s7-discussion';
import finM15S8Discussion from '../discussions/finance/fin-m15-s8-discussion';
import finM15S9Discussion from '../discussions/finance/fin-m15-s9-discussion';
import finM15S10Discussion from '../discussions/finance/fin-m15-s10-discussion';
import finM15S11Discussion from '../discussions/finance/fin-m15-s11-discussion';
import finM15S12Discussion from '../discussions/finance/fin-m15-s12-discussion';
import finM15S13Discussion from '../discussions/finance/fin-m15-s13-discussion';
import finM15S14Discussion from '../discussions/finance/fin-m15-s14-discussion';
import finM15S15Discussion from '../discussions/finance/fin-m15-s15-discussion';
import finM15S16Discussion from '../discussions/finance/fin-m15-s16-discussion';

// Quiz imports
import finM15S1Quiz from '../quizzes/finance/fin-m15-s1-quiz';
import finM15S2Quiz from '../quizzes/finance/fin-m15-s2-quiz';
import finM15S3Quiz from '../quizzes/finance/fin-m15-s3-quiz';
import finM15S4Quiz from '../quizzes/finance/fin-m15-s4-quiz';
import finM15S5Quiz from '../quizzes/finance/fin-m15-s5-quiz';
import finM15S6Quiz from '../quizzes/finance/fin-m15-s6-quiz';
import finM15S7Quiz from '../quizzes/finance/fin-m15-s7-quiz';
import finM15S8Quiz from '../quizzes/finance/fin-m15-s8-quiz';
import finM15S9Quiz from '../quizzes/finance/fin-m15-s9-quiz';
import finM15S10Quiz from '../quizzes/finance/fin-m15-s10-quiz';
import finM15S11Quiz from '../quizzes/finance/fin-m15-s11-quiz';
import finM15S12Quiz from '../quizzes/finance/fin-m15-s12-quiz';
import finM15S13Quiz from '../quizzes/finance/fin-m15-s13-quiz';
import finM15S14Quiz from '../quizzes/finance/fin-m15-s14-quiz';
import finM15S15Quiz from '../quizzes/finance/fin-m15-s15-quiz';
import finM15S16Quiz from '../quizzes/finance/fin-m15-s16-quiz';

export const riskManagementPortfolioSystemsModule: Module = {
  id: 'risk-management-portfolio-systems',
  title: 'Risk Management & Portfolio Systems',
  description:
    'Master institutional risk management from fundamentals to production systems managing $10T+. Learn VaR methods (Historical, Parametric, Monte Carlo), CVaR, stress testing, and comprehensive risk management across market, credit, operational, and liquidity risks. Understand real-time monitoring, reporting, regulatory requirements (Basel III, Dodd-Frank), and build production-grade risk platforms. Study BlackRock Aladdin architecture and implement a complete risk management system capstone project.',
  icon: '🛡️',
  keyTakeaways: [
    'Understand risk taxonomy: market, credit, operational, liquidity, and their interactions',
    'Master VaR calculation: Historical simulation, Parametric, Monte Carlo (accuracy vs speed)',
    'Calculate CVaR (Expected Shortfall) for tail risk measurement and coherent risk metrics',
    'Conduct stress testing: historical scenarios, hypothetical stress, reverse stress testing',
    'Manage market risk: Greeks (Delta, Gamma, Vega), VaR, stress testing, backtesting, regulatory capital',
    'Manage credit risk: PD/LGD/EAD models, Credit VaR, CDS, CVA/DVA, counterparty risk',
    'Understand operational risk: people/process/systems/external, Basel SMA vs AMA failure',
    'Monitor liquidity risk: LCR, NSFR, funding vs market liquidity, contingency funding plans',
    'Perform risk attribution: Component VaR, Marginal VaR, Incremental VaR, factor-based attribution',
    'Implement risk budgeting: Risk parity, marginal contribution to risk, capital allocation',
    'Manage margin/collateral: IM vs VM, ISDA/CSA, UMR requirements, cheapest-to-deliver optimization',
    'Set position limits: Hard vs soft limits, pre-trade checks, kill switches, breach escalation',
    'Build real-time monitoring: Sub-second VaR, WebSocket dashboards, unexplained P&L detection',
    'Design risk reporting: Board vs regulatory vs management reports, exception-based reporting',
    'Analyze Aladdin architecture: $10T+ platform, unified data model, network effects, competitive moat',
    'Build risk platform: Real-time position tracking, VaR engine, limit monitoring, audit trail',
    'Apply regulatory frameworks: Basel III (VaR, Stressed VaR, backtesting), Dodd-Frank, MiFID II',
    'Implement production systems: <100ms pre-trade checks, hierarchical aggregation, distributed compute',
  ],
  sections: [
    {
      ...riskManagementFundamentals,
      discussion: finM15S1Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S1Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...valueAtRiskMethods,
      discussion: finM15S2Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S2Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...conditionalValueAtRisk,
      discussion: finM15S3Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S3Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...stressTestingScenarioAnalysis,
      discussion: finM15S4Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S4Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...marketRiskManagement,
      discussion: finM15S5Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S5Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...creditRiskManagement,
      discussion: finM15S6Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S6Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...operationalRisk,
      discussion: finM15S7Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S7Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...liquidityRisk,
      discussion: finM15S8Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S8Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...riskAttributionAnalysis,
      discussion: finM15S9Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S9Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...riskBudgeting,
      discussion: finM15S10Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S10Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...marginCollateralManagement,
      discussion: finM15S11Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S11Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...positionLimitsRiskLimits,
      discussion: finM15S12Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S12Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...realTimeRiskMonitoring,
      discussion: finM15S13Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S13Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...riskReportingDashboards,
      discussion: finM15S14Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S14Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...blackrockAladdinArchitecture,
      discussion: finM15S15Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S15Quiz.questions as unknown as QuizQuestion[],
    },
    {
      ...riskManagementPlatformProject,
      discussion: finM15S16Discussion.questions as unknown as DiscussionItem[],
      quiz: finM15S16Quiz.questions as unknown as QuizQuestion[],
    },
  ],
};
