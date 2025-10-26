import { Module } from '../../types';

// Import sections
import { tradingSystemArchitecture } from '../sections/building-trading-infrastructure/trading-system-architecture';
import { orderManagementSystem } from '../sections/building-trading-infrastructure/order-management-system';
import { executionManagementSystem } from '../sections/building-trading-infrastructure/execution-management-system';
import { fixProtocolDeepDive } from '../sections/building-trading-infrastructure/fix-protocol-deep-dive';
import { smartOrderRouting } from '../sections/building-trading-infrastructure/smart-order-routing';
import { positionTrackingReconciliation } from '../sections/building-trading-infrastructure/position-tracking-reconciliation';
import { pnlCalculation } from '../sections/building-trading-infrastructure/pnl-calculation';
import { tradeReconciliation } from '../sections/building-trading-infrastructure/trade-reconciliation';
import { lowLatencyProgramming } from '../sections/building-trading-infrastructure/low-latency-programming';
import { messageQueuesTrading } from '../sections/building-trading-infrastructure/message-queues-trading';
import { databaseDesignTrading } from '../sections/building-trading-infrastructure/database-design-trading';
import { systemMonitoringAlerting } from '../sections/building-trading-infrastructure/system-monitoring-alerting';
import { disasterRecoveryFailover } from '../sections/building-trading-infrastructure/disaster-recovery-failover';
import { productionDeployment } from '../sections/building-trading-infrastructure/production-deployment';
import { completeTradingSystem } from '../sections/building-trading-infrastructure/complete-trading-system';

// Import quizzes
import { tradingSystemArchitectureQuiz } from '../quizzes/building-trading-infrastructure/trading-system-architecture';
import { orderManagementSystemQuiz } from '../quizzes/building-trading-infrastructure/order-management-system';
import { executionManagementSystemQuiz } from '../quizzes/building-trading-infrastructure/execution-management-system';
import { fixProtocolDeepDiveQuiz } from '../quizzes/building-trading-infrastructure/fix-protocol-deep-dive';
import { smartOrderRoutingQuiz } from '../quizzes/building-trading-infrastructure/smart-order-routing';
import { positionTrackingReconciliationQuiz } from '../quizzes/building-trading-infrastructure/position-tracking-reconciliation';
import { pnlCalculationQuiz } from '../quizzes/building-trading-infrastructure/pnl-calculation';
import { tradeReconciliationQuiz } from '../quizzes/building-trading-infrastructure/trade-reconciliation';
import { lowLatencyProgrammingQuiz } from '../quizzes/building-trading-infrastructure/low-latency-programming';
import { messageQueuesTradingQuiz } from '../quizzes/building-trading-infrastructure/message-queues-trading';
import { databaseDesignTradingQuiz } from '../quizzes/building-trading-infrastructure/database-design-trading';
import { systemMonitoringAlertingQuiz } from '../quizzes/building-trading-infrastructure/system-monitoring-alerting';
import { disasterRecoveryFailoverQuiz } from '../quizzes/building-trading-infrastructure/disaster-recovery-failover';
import { productionDeploymentQuiz } from '../quizzes/building-trading-infrastructure/production-deployment';
import { completeTradingSystemQuiz } from '../quizzes/building-trading-infrastructure/complete-trading-system';

// Import multiple choice
import { tradingSystemArchitectureMC } from '../multiple-choice/building-trading-infrastructure/trading-system-architecture';
import { orderManagementSystemMC } from '../multiple-choice/building-trading-infrastructure/order-management-system';
import { executionManagementSystemMC } from '../multiple-choice/building-trading-infrastructure/execution-management-system';
import { fixProtocolDeepDiveMC } from '../multiple-choice/building-trading-infrastructure/fix-protocol-deep-dive';
import { smartOrderRoutingMC } from '../multiple-choice/building-trading-infrastructure/smart-order-routing';
import { positionTrackingReconciliationMC } from '../multiple-choice/building-trading-infrastructure/position-tracking-reconciliation';
import { pnlCalculationMC } from '../multiple-choice/building-trading-infrastructure/pnl-calculation';
import { tradeReconciliationMC } from '../multiple-choice/building-trading-infrastructure/trade-reconciliation';
import { lowLatencyProgrammingMC } from '../multiple-choice/building-trading-infrastructure/low-latency-programming';
import { messageQueuesTradingMC } from '../multiple-choice/building-trading-infrastructure/message-queues-trading';
import { databaseDesignTradingMC } from '../multiple-choice/building-trading-infrastructure/database-design-trading';
import { systemMonitoringAlertingMC } from '../multiple-choice/building-trading-infrastructure/system-monitoring-alerting';
import { disasterRecoveryFailoverMC } from '../multiple-choice/building-trading-infrastructure/disaster-recovery-failover';
import { productionDeploymentMC } from '../multiple-choice/building-trading-infrastructure/production-deployment';
import { completeTradingSystemMC } from '../multiple-choice/building-trading-infrastructure/complete-trading-system';

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

export const buildingTradingInfrastructureModule: Module = {
    id: 'building-trading-infrastructure',
    title: 'Building Trading Infrastructure',
    description:
        'Build production-grade trading systems: OMS, EMS, FIX Protocol, position tracking, P&L calculation, and deployment',
    sections: [
        {
            id: tradingSystemArchitecture.id,
            title: tradingSystemArchitecture.title,
            content: tradingSystemArchitecture.content,
            quiz: tradingSystemArchitectureQuiz.map(transformQuiz),
            multipleChoice: tradingSystemArchitectureMC.map(transformMC),
        },
        {
            id: orderManagementSystem.id,
            title: orderManagementSystem.title,
            content: orderManagementSystem.content,
            quiz: orderManagementSystemQuiz.map(transformQuiz),
            multipleChoice: orderManagementSystemMC.map(transformMC),
        },
        {
            id: executionManagementSystem.id,
            title: executionManagementSystem.title,
            content: executionManagementSystem.content,
            quiz: executionManagementSystemQuiz.map(transformQuiz),
            multipleChoice: executionManagementSystemMC.map(transformMC),
        },
        {
            id: fixProtocolDeepDive.id,
            title: fixProtocolDeepDive.title,
            content: fixProtocolDeepDive.content,
            quiz: fixProtocolDeepDiveQuiz.map(transformQuiz),
            multipleChoice: fixProtocolDeepDiveMC.map(transformMC),
        },
        {
            id: smartOrderRouting.id,
            title: smartOrderRouting.title,
            content: smartOrderRouting.content,
            quiz: smartOrderRoutingQuiz.map(transformQuiz),
            multipleChoice: smartOrderRoutingMC.map(transformMC),
        },
        {
            id: positionTrackingReconciliation.id,
            title: positionTrackingReconciliation.title,
            content: positionTrackingReconciliation.content,
            quiz: positionTrackingReconciliationQuiz.map(transformQuiz),
            multipleChoice: positionTrackingReconciliationMC.map(transformMC),
        },
        {
            id: pnlCalculation.id,
            title: pnlCalculation.title,
            content: pnlCalculation.content,
            quiz: pnlCalculationQuiz.map(transformQuiz),
            multipleChoice: pnlCalculationMC.map(transformMC),
        },
        {
            id: tradeReconciliation.id,
            title: tradeReconciliation.title,
            content: tradeReconciliation.content,
            quiz: tradeReconciliationQuiz.map(transformQuiz),
            multipleChoice: tradeReconciliationMC.map(transformMC),
        },
        {
            id: lowLatencyProgramming.id,
            title: lowLatencyProgramming.title,
            content: lowLatencyProgramming.content,
            quiz: lowLatencyProgrammingQuiz.map(transformQuiz),
            multipleChoice: lowLatencyProgrammingMC.map(transformMC),
        },
        {
            id: messageQueuesTrading.id,
            title: messageQueuesTrading.title,
            content: messageQueuesTrading.content,
            quiz: messageQueuesTradingQuiz.map(transformQuiz),
            multipleChoice: messageQueuesTradingMC.map(transformMC),
        },
        {
            id: databaseDesignTrading.id,
            title: databaseDesignTrading.title,
            content: databaseDesignTrading.content,
            quiz: databaseDesignTradingQuiz.map(transformQuiz),
            multipleChoice: databaseDesignTradingMC.map(transformMC),
        },
        {
            id: systemMonitoringAlerting.id,
            title: systemMonitoringAlerting.title,
            content: systemMonitoringAlerting.content,
            quiz: systemMonitoringAlertingQuiz.map(transformQuiz),
            multipleChoice: systemMonitoringAlertingMC.map(transformMC),
        },
        {
            id: disasterRecoveryFailover.id,
            title: disasterRecoveryFailover.title,
            content: disasterRecoveryFailover.content,
            quiz: disasterRecoveryFailoverQuiz.map(transformQuiz),
            multipleChoice: disasterRecoveryFailoverMC.map(transformMC),
        },
        {
            id: productionDeployment.id,
            title: productionDeployment.title,
            content: productionDeployment.content,
            quiz: productionDeploymentQuiz.map(transformQuiz),
            multipleChoice: productionDeploymentMC.map(transformMC),
        },
        {
            id: completeTradingSystem.id,
            title: completeTradingSystem.title,
            content: completeTradingSystem.content,
            quiz: completeTradingSystemQuiz.map(transformQuiz),
            multipleChoice: completeTradingSystemMC.map(transformMC),
        },
    ],
    quizzes: [
        ...tradingSystemArchitectureQuiz.map(transformQuiz),
        ...orderManagementSystemQuiz.map(transformQuiz),
        ...executionManagementSystemQuiz.map(transformQuiz),
        ...fixProtocolDeepDiveQuiz.map(transformQuiz),
        ...smartOrderRoutingQuiz.map(transformQuiz),
        ...positionTrackingReconciliationQuiz.map(transformQuiz),
        ...pnlCalculationQuiz.map(transformQuiz),
        ...tradeReconciliationQuiz.map(transformQuiz),
        ...lowLatencyProgrammingQuiz.map(transformQuiz),
        ...messageQueuesTradingQuiz.map(transformQuiz),
        ...databaseDesignTradingQuiz.map(transformQuiz),
        ...systemMonitoringAlertingQuiz.map(transformQuiz),
        ...disasterRecoveryFailoverQuiz.map(transformQuiz),
        ...productionDeploymentQuiz.map(transformQuiz),
        ...completeTradingSystemQuiz.map(transformQuiz),
    ],
    multipleChoiceQuestions: [
        ...tradingSystemArchitectureMC.map(transformMC),
        ...orderManagementSystemMC.map(transformMC),
        ...executionManagementSystemMC.map(transformMC),
        ...fixProtocolDeepDiveMC.map(transformMC),
        ...smartOrderRoutingMC.map(transformMC),
        ...positionTrackingReconciliationMC.map(transformMC),
        ...pnlCalculationMC.map(transformMC),
        ...tradeReconciliationMC.map(transformMC),
        ...lowLatencyProgrammingMC.map(transformMC),
        ...messageQueuesTradingMC.map(transformMC),
        ...databaseDesignTradingMC.map(transformMC),
        ...systemMonitoringAlertingMC.map(transformMC),
        ...disasterRecoveryFailoverMC.map(transformMC),
        ...productionDeploymentMC.map(transformMC),
        ...completeTradingSystemMC.map(transformMC),
    ],
};

