import { Module } from '../../types';

// Import sections
import { financialModelingFundamentals } from '@/lib/content/sections/financial-modeling-valuation/financial-modeling-fundamentals';
import { threeStatementModel } from '@/lib/content/sections/financial-modeling-valuation/three-statement-model';
import { dcfModel } from '@/lib/content/sections/financial-modeling-valuation/dcf-model';
import { comparableCompanyAnalysis } from '@/lib/content/sections/financial-modeling-valuation/comparable-company-analysis';
import { precedentTransactions } from '@/lib/content/sections/financial-modeling-valuation/precedent-transactions';
import { lboModel } from '@/lib/content/sections/financial-modeling-valuation/lbo-model';
import { maModel } from '@/lib/content/sections/financial-modeling-valuation/ma-model';
import { sensitivityScenarioAnalysis } from '@/lib/content/sections/financial-modeling-valuation/sensitivity-scenario-analysis';
import { monteCarloValuation } from '@/lib/content/sections/financial-modeling-valuation/monte-carlo-valuation';
import { realOptionsValuation } from '@/lib/content/sections/financial-modeling-valuation/real-options-valuation';
import { dividendDiscountModel } from '@/lib/content/sections/financial-modeling-valuation/dividend-discount-model';
import { sumOfPartsValuation } from '@/lib/content/sections/financial-modeling-valuation/sum-of-parts-valuation';
import { automatedModelGeneration } from '@/lib/content/sections/financial-modeling-valuation/automated-model-generation';
import { valuationPlatformProject } from '@/lib/content/sections/financial-modeling-valuation/valuation-platform-project';

// Import quizzes
import { financialModelingFundamentalsQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/financial-modeling-fundamentals';
import { threeStatementModelQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/three-statement-model';
import { dcfModelQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/dcf-model';
import { comparableCompanyAnalysisQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/comparable-company-analysis';
import { precedentTransactionsQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/precedent-transactions';
import { lboModelQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/lbo-model';
import { maModelQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/ma-model';
import { sensitivityScenarioAnalysisQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/sensitivity-scenario-analysis';
import { monteCarloValuationQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/monte-carlo-valuation';
import { realOptionsValuationQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/real-options-valuation';
import { dividendDiscountModelQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/dividend-discount-model';
import { sumOfPartsValuationQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/sum-of-parts-valuation';
import { automatedModelGenerationQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/automated-model-generation';
import { valuationPlatformProjectQuiz } from '@/lib/content/quizzes/financial-modeling-valuation/valuation-platform-project';

// Import multiple choice
import { financialModelingFundamentalsMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/financial-modeling-fundamentals';
import { threeStatementModelMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/three-statement-model';
import { dcfModelMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/dcf-model';
import { comparableCompanyAnalysisMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/comparable-company-analysis';
import { precedentTransactionsMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/precedent-transactions';
import { lboModelMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/lbo-model';
import { maModelMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/ma-model';
import { sensitivityScenarioAnalysisMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/sensitivity-scenario-analysis';
import { monteCarloValuationMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/monte-carlo-valuation';
import { realOptionsValuationMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/real-options-valuation';
import { dividendDiscountModelMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/dividend-discount-model';
import { sumOfPartsValuationMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/sum-of-parts-valuation';
import { automatedModelGenerationMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/automated-model-generation';
import { valuationPlatformProjectMultipleChoice } from '@/lib/content/multiple-choice/financial-modeling-valuation/valuation-platform-project';

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

export const financialModelingValuationModule: Module = {
    id: 'financial-modeling-valuation',
    title: 'Financial Modeling & Valuation',
    description:
        'Master DCF, comps, LBO models, and valuation techniques used by investment banks and PE firms',
    sections: [
        {
            id: financialModelingFundamentals.id,
            title: financialModelingFundamentals.title,
            content: financialModelingFundamentals.content,
            quiz: financialModelingFundamentalsQuiz.map(transformQuiz),
            multipleChoice: financialModelingFundamentalsMultipleChoice.map(transformMC),
        },
        {
            id: threeStatementModel.id,
            title: threeStatementModel.title,
            content: threeStatementModel.content,
            quiz: threeStatementModelQuiz.map(transformQuiz),
            multipleChoice: threeStatementModelMultipleChoice.map(transformMC),
        },
        {
            id: dcfModel.id,
            title: dcfModel.title,
            content: dcfModel.content,
            quiz: dcfModelQuiz.map(transformQuiz),
            multipleChoice: dcfModelMultipleChoice.map(transformMC),
        },
        {
            id: comparableCompanyAnalysis.id,
            title: comparableCompanyAnalysis.title,
            content: comparableCompanyAnalysis.content,
            quiz: comparableCompanyAnalysisQuiz.map(transformQuiz),
            multipleChoice: comparableCompanyAnalysisMultipleChoice.map(transformMC),
        },
        {
            id: precedentTransactions.id,
            title: precedentTransactions.title,
            content: precedentTransactions.content,
            quiz: precedentTransactionsQuiz.map(transformQuiz),
            multipleChoice: precedentTransactionsMultipleChoice.map(transformMC),
        },
        {
            id: lboModel.id,
            title: lboModel.title,
            content: lboModel.content,
            quiz: lboModelQuiz.map(transformQuiz),
            multipleChoice: lboModelMultipleChoice.map(transformMC),
        },
        {
            id: maModel.id,
            title: maModel.title,
            content: maModel.content,
            quiz: maModelQuiz.map(transformQuiz),
            multipleChoice: maModelMultipleChoice.map(transformMC),
        },
        {
            id: sensitivityScenarioAnalysis.id,
            title: sensitivityScenarioAnalysis.title,
            content: sensitivityScenarioAnalysis.content,
            quiz: sensitivityScenarioAnalysisQuiz.map(transformQuiz),
            multipleChoice: sensitivityScenarioAnalysisMultipleChoice.map(transformMC),
        },
        {
            id: monteCarloValuation.id,
            title: monteCarloValuation.title,
            content: monteCarloValuation.content,
            quiz: monteCarloValuationQuiz.map(transformQuiz),
            multipleChoice: monteCarloValuationMultipleChoice.map(transformMC),
        },
        {
            id: realOptionsValuation.id,
            title: realOptionsValuation.title,
            content: realOptionsValuation.content,
            quiz: realOptionsValuationQuiz.map(transformQuiz),
            multipleChoice: realOptionsValuationMultipleChoice.map(transformMC),
        },
        {
            id: dividendDiscountModel.id,
            title: dividendDiscountModel.title,
            content: dividendDiscountModel.content,
            quiz: dividendDiscountModelQuiz.map(transformQuiz),
            multipleChoice: dividendDiscountModelMultipleChoice.map(transformMC),
        },
        {
            id: sumOfPartsValuation.id,
            title: sumOfPartsValuation.title,
            content: sumOfPartsValuation.content,
            quiz: sumOfPartsValuationQuiz.map(transformQuiz),
            multipleChoice: sumOfPartsValuationMultipleChoice.map(transformMC),
        },
        {
            id: automatedModelGeneration.id,
            title: automatedModelGeneration.title,
            content: automatedModelGeneration.content,
            quiz: automatedModelGenerationQuiz.map(transformQuiz),
            multipleChoice: automatedModelGenerationMultipleChoice.map(transformMC),
        },
        {
            id: valuationPlatformProject.id,
            title: valuationPlatformProject.title,
            content: valuationPlatformProject.content,
            quiz: valuationPlatformProjectQuiz.map(transformQuiz),
            multipleChoice: valuationPlatformProjectMultipleChoice.map(transformMC),
        },
    ],
    quizzes: [
        ...financialModelingFundamentalsQuiz.map(transformQuiz),
        ...threeStatementModelQuiz.map(transformQuiz),
        ...dcfModelQuiz.map(transformQuiz),
        ...comparableCompanyAnalysisQuiz.map(transformQuiz),
        ...precedentTransactionsQuiz.map(transformQuiz),
        ...lboModelQuiz.map(transformQuiz),
        ...maModelQuiz.map(transformQuiz),
        ...sensitivityScenarioAnalysisQuiz.map(transformQuiz),
        ...monteCarloValuationQuiz.map(transformQuiz),
        ...realOptionsValuationQuiz.map(transformQuiz),
        ...dividendDiscountModelQuiz.map(transformQuiz),
        ...sumOfPartsValuationQuiz.map(transformQuiz),
        ...automatedModelGenerationQuiz.map(transformQuiz),
        ...valuationPlatformProjectQuiz.map(transformQuiz),
    ],
    multipleChoiceQuestions: [
        ...financialModelingFundamentalsMultipleChoice.map(transformMC),
        ...threeStatementModelMultipleChoice.map(transformMC),
        ...dcfModelMultipleChoice.map(transformMC),
        ...comparableCompanyAnalysisMultipleChoice.map(transformMC),
        ...precedentTransactionsMultipleChoice.map(transformMC),
        ...lboModelMultipleChoice.map(transformMC),
        ...maModelMultipleChoice.map(transformMC),
        ...sensitivityScenarioAnalysisMultipleChoice.map(transformMC),
        ...monteCarloValuationMultipleChoice.map(transformMC),
        ...realOptionsValuationMultipleChoice.map(transformMC),
        ...dividendDiscountModelMultipleChoice.map(transformMC),
        ...sumOfPartsValuationMultipleChoice.map(transformMC),
        ...automatedModelGenerationMultipleChoice.map(transformMC),
        ...valuationPlatformProjectMultipleChoice.map(transformMC),
    ],
};

