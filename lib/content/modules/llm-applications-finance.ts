/**
 * Module: LLM Applications in Finance
 * Module 18 of ML & AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { financialDocumentAnalysis } from '../sections/llm-applications-finance/financial-document-analysis';
import { earningsCallAnalysis } from '../sections/llm-applications-finance/earnings-call-analysis';
import { financialNewsAnalysis } from '../sections/llm-applications-finance/financial-news-analysis';
import { automatedReportGeneration } from '../sections/llm-applications-finance/automated-report-generation';
import { tradingSignalGeneration } from '../sections/llm-applications-finance/trading-signal-generation';
import { riskAssessmentLLMs } from '../sections/llm-applications-finance/risk-assessment-llms';
import { marketResearchAutomation } from '../sections/llm-applications-finance/market-research-automation';
import { conversationalTradingAssistants } from '../sections/llm-applications-finance/conversational-trading-assistants';
import { llmBacktestingStrategyDevelopment } from '../sections/llm-applications-finance/llm-backtesting-strategy-development';
import { regulatoryComplianceMonitoring } from '../sections/llm-applications-finance/regulatory-compliance-monitoring';

// Quiz imports
import { financialDocumentAnalysisQuiz } from '../quizzes/llm-applications-finance/financial-document-analysis';
import { earningsCallAnalysisQuiz } from '../quizzes/llm-applications-finance/earnings-call-analysis';
import { financialNewsAnalysisQuiz } from '../quizzes/llm-applications-finance/financial-news-analysis';
import { automatedReportGenerationQuiz } from '../quizzes/llm-applications-finance/automated-report-generation';
import { tradingSignalGenerationQuiz } from '../quizzes/llm-applications-finance/trading-signal-generation';
import { riskAssessmentLLMsQuiz } from '../quizzes/llm-applications-finance/risk-assessment-llms';
import { marketResearchAutomationQuiz } from '../quizzes/llm-applications-finance/market-research-automation';
import { conversationalTradingAssistantsQuiz } from '../quizzes/llm-applications-finance/conversational-trading-assistants';
import { llmBacktestingStrategyDevelopmentQuiz } from '../quizzes/llm-applications-finance/llm-backtesting-strategy-development';
import { regulatoryComplianceMonitoringQuiz } from '../quizzes/llm-applications-finance/regulatory-compliance-monitoring';

// Multiple choice imports
import { financialDocumentAnalysisMultipleChoice } from '../multiple-choice/llm-applications-finance/financial-document-analysis';
import { earningsCallAnalysisMultipleChoice } from '../multiple-choice/llm-applications-finance/earnings-call-analysis';
import { financialNewsAnalysisMultipleChoice } from '../multiple-choice/llm-applications-finance/financial-news-analysis';
import { automatedReportGenerationMultipleChoice } from '../multiple-choice/llm-applications-finance/automated-report-generation';
import { tradingSignalGenerationMultipleChoice } from '../multiple-choice/llm-applications-finance/trading-signal-generation';
import { riskAssessmentLLMsMultipleChoice } from '../multiple-choice/llm-applications-finance/risk-assessment-llms';
import { marketResearchAutomationMultipleChoice } from '../multiple-choice/llm-applications-finance/market-research-automation';
import { conversationalTradingAssistantsMultipleChoice } from '../multiple-choice/llm-applications-finance/conversational-trading-assistants';
import { llmBacktestingStrategyDevelopmentMultipleChoice } from '../multiple-choice/llm-applications-finance/llm-backtesting-strategy-development';
import { regulatoryComplianceMonitoringMultipleChoice } from '../multiple-choice/llm-applications-finance/regulatory-compliance-monitoring';

// Helper to transform quiz format from llm-applications-finance format to standard format
const transformQuiz = (
  quiz: { id: number; question: string; expectedAnswer: string }[],
) => {
  return quiz.map((q) => ({
    id: q.id.toString(),
    question: q.question,
    sampleAnswer: q.expectedAnswer,
    keyPoints: [], // LLM quizzes don't have keyPoints, so we'll use empty array
  }));
};

// Helper to transform multiple choice format
const transformMC = (
  mc: {
    id: number;
    question: string;
    options: string[];
    correctAnswer: number;
    explanation: string;
  }[],
) => {
  return mc.map((q) => ({
    id: q.id.toString(),
    question: q.question,
    options: q.options,
    correctAnswer: q.correctAnswer,
    explanation: q.explanation,
  }));
};

export const llmApplicationsFinanceModule: Module = {
  id: 'ml-ai-llm-applications-finance',
  title: 'LLM Applications in Finance',
  description:
    'Master production LLM applications transforming finance and trading. Build systems for financial document analysis (10-K/Q, earnings), automated earnings call analysis with sentiment and speaker detection, real-time financial news processing at scale, automated portfolio report generation, trading signal generation from multiple data sources, credit and geopolitical risk assessment, market research automation, conversational trading assistants with voice commands, LLM-powered strategy development and backtesting, and regulatory compliance monitoring. Deploy production systems handling millions in trading decisions with proper evaluation, risk controls, and compliance.',
  icon: '💼',
  keyTakeaways: [
    'Analyze financial documents (10-K, 10-Q, 8-K) with LLMs to extract insights, detect tone shifts, and identify risks',
    'Build automated earnings call analysis systems with sentiment detection and Q&A pattern recognition',
    'Process financial news at scale with deduplication, source weighting, and real-time signal generation',
    'Generate personalized automated portfolio reports with compliance and hallucination prevention',
    'Create trading signal systems combining technical, fundamental, sentiment, and news data',
    'Implement credit risk assessment using qualitative management discussion analysis',
    'Deploy geopolitical and counterparty risk monitoring with early warning systems',
    'Automate competitive analysis, industry research, and due diligence workflows',
    'Build conversational trading assistants with voice commands, safety mechanisms, and compliance',
    'Generate and backtest trading strategies from natural language descriptions',
    'Use LLMs for strategy explanation, parameter optimization, and walk-forward analysis',
    'Implement regulatory compliance monitoring for communications and policy changes',
    'Handle compliance, audit trails, and suitability requirements in automated systems',
    'Deploy production systems with proper risk controls, confidence calibration, and human oversight',
    'Manage costs, latency, and reliability in high-stakes financial applications',
    'Build complete financial LLM systems from research to production deployment',
  ],
  sections: [
    {
      ...financialDocumentAnalysis,
      quiz: transformQuiz(financialDocumentAnalysisQuiz.questions),
      multipleChoice: transformMC(
        financialDocumentAnalysisMultipleChoice.questions,
      ),
    },
    {
      ...earningsCallAnalysis,
      quiz: transformQuiz(earningsCallAnalysisQuiz.questions),
      multipleChoice: transformMC(earningsCallAnalysisMultipleChoice.questions),
    },
    {
      ...financialNewsAnalysis,
      quiz: transformQuiz(financialNewsAnalysisQuiz.questions),
      multipleChoice: transformMC(
        financialNewsAnalysisMultipleChoice.questions,
      ),
    },
    {
      ...automatedReportGeneration,
      quiz: transformQuiz(automatedReportGenerationQuiz.questions),
      multipleChoice: transformMC(
        automatedReportGenerationMultipleChoice.questions,
      ),
    },
    {
      ...tradingSignalGeneration,
      quiz: transformQuiz(tradingSignalGenerationQuiz.questions),
      multipleChoice: transformMC(
        tradingSignalGenerationMultipleChoice.questions,
      ),
    },
    {
      ...riskAssessmentLLMs,
      quiz: transformQuiz(riskAssessmentLLMsQuiz.questions),
      multipleChoice: transformMC(riskAssessmentLLMsMultipleChoice.questions),
    },
    {
      ...marketResearchAutomation,
      quiz: transformQuiz(marketResearchAutomationQuiz.questions),
      multipleChoice: transformMC(
        marketResearchAutomationMultipleChoice.questions,
      ),
    },
    {
      ...conversationalTradingAssistants,
      quiz: transformQuiz(conversationalTradingAssistantsQuiz.questions),
      multipleChoice: transformMC(
        conversationalTradingAssistantsMultipleChoice.questions,
      ),
    },
    {
      ...llmBacktestingStrategyDevelopment,
      quiz: transformQuiz(llmBacktestingStrategyDevelopmentQuiz.questions),
      multipleChoice: transformMC(
        llmBacktestingStrategyDevelopmentMultipleChoice.questions,
      ),
    },
    {
      ...regulatoryComplianceMonitoring,
      quiz: transformQuiz(regulatoryComplianceMonitoringQuiz.questions),
      multipleChoice: transformMC(
        regulatoryComplianceMonitoringMultipleChoice.questions,
      ),
    },
  ],
};
