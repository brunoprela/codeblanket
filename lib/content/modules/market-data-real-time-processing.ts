import { Module } from '../../types';

// Import sections
import { marketDataFundamentals } from '@/lib/content/sections/market-data-real-time-processing/market-data-fundamentals';
import { dataFeedProtocols } from '@/lib/content/sections/market-data-real-time-processing/data-feed-protocols';
import { tickDataProcessing } from '@/lib/content/sections/market-data-real-time-processing/tick-data-processing';
import { ohlcvBarConstruction } from '@/lib/content/sections/market-data-real-time-processing/ohlcv-bar-construction';
import { levelData } from '@/lib/content/sections/market-data-real-time-processing/level-data';
import { marketDataVendorsApis } from '@/lib/content/sections/market-data-real-time-processing/market-data-vendors-apis';
import { realTimeDataPipelines } from '@/lib/content/sections/market-data-real-time-processing/real-time-data-pipelines';
import { dataNormalization } from '@/lib/content/sections/market-data-real-time-processing/data-normalization';
import { timestampManagement } from '@/lib/content/sections/market-data-real-time-processing/timestamp-management';
import { marketDataStorage } from '@/lib/content/sections/market-data-real-time-processing/market-data-storage';
import { dataQualityValidation } from '@/lib/content/sections/market-data-real-time-processing/data-quality-validation';
import { lowLatencyProcessing } from '@/lib/content/sections/market-data-real-time-processing/low-latency-processing';
import { realTimePlatformProject } from '@/lib/content/sections/market-data-real-time-processing/real-time-platform-project';

// Import quizzes
import { marketDataFundamentalsQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/market-data-fundamentals';
import { dataFeedProtocolsQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/data-feed-protocols';
import { tickDataProcessingQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/tick-data-processing';
import { ohlcvBarConstructionQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/ohlcv-bar-construction';
import { levelDataQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/level-data';
import { marketDataVendorsApisQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/market-data-vendors-apis';
import { realTimeDataPipelinesQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/real-time-data-pipelines';
import { dataNormalizationQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/data-normalization';
import { timestampManagementQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/timestamp-management';
import { marketDataStorageQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/market-data-storage';
import { dataQualityValidationQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/data-quality-validation';
import { lowLatencyProcessingQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/low-latency-processing';
import { realTimePlatformProjectQuiz } from '@/lib/content/quizzes/market-data-real-time-processing/real-time-platform-project';

// Import multiple choice
import { marketDataFundamentalsMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/market-data-fundamentals';
import { dataFeedProtocolsMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/data-feed-protocols';
import { tickDataProcessingMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/tick-data-processing';
import { ohlcvBarConstructionMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/ohlcv-bar-construction';
import { levelDataMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/level-data';
import { marketDataVendorsApisMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/market-data-vendors-apis';
import { realTimeDataPipelinesMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/real-time-data-pipelines';
import { dataNormalizationMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/data-normalization';
import { timestampManagementMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/timestamp-management';
import { marketDataStorageMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/market-data-storage';
import { dataQualityValidationMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/data-quality-validation';
import { lowLatencyProcessingMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/low-latency-processing';
import { realTimePlatformProjectMultipleChoice } from '@/lib/content/multiple-choice/market-data-real-time-processing/real-time-platform-project';

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

export const marketDataRealTimeProcessingModule: Module = {
    id: 'market-data-real-time-processing',
    title: 'Market Data & Real-Time Processing',
    description:
        'Master real-time market data processing, low-latency infrastructure, and production data pipelines for financial markets',
    sections: [
        {
            id: marketDataFundamentals.id,
            title: marketDataFundamentals.title,
            content: marketDataFundamentals.content,
            quiz: marketDataFundamentalsQuiz.map(transformQuiz),
            multipleChoice: marketDataFundamentalsMultipleChoice.map(transformMC),
        },
        {
            id: dataFeedProtocols.id,
            title: dataFeedProtocols.title,
            content: dataFeedProtocols.content,
            quiz: dataFeedProtocolsQuiz.map(transformQuiz),
            multipleChoice: dataFeedProtocolsMultipleChoice.map(transformMC),
        },
        {
            id: tickDataProcessing.id,
            title: tickDataProcessing.title,
            content: tickDataProcessing.content,
            quiz: tickDataProcessingQuiz.map(transformQuiz),
            multipleChoice: tickDataProcessingMultipleChoice.map(transformMC),
        },
        {
            id: ohlcvBarConstruction.id,
            title: ohlcvBarConstruction.title,
            content: ohlcvBarConstruction.content,
            quiz: ohlcvBarConstructionQuiz.map(transformQuiz),
            multipleChoice: ohlcvBarConstructionMultipleChoice.map(transformMC),
        },
        {
            id: levelData.id,
            title: levelData.title,
            content: levelData.content,
            quiz: levelDataQuiz.map(transformQuiz),
            multipleChoice: levelDataMultipleChoice.map(transformMC),
        },
        {
            id: marketDataVendorsApis.id,
            title: marketDataVendorsApis.title,
            content: marketDataVendorsApis.content,
            quiz: marketDataVendorsApisQuiz.map(transformQuiz),
            multipleChoice: marketDataVendorsApisMultipleChoice.map(transformMC),
        },
        {
            id: realTimeDataPipelines.id,
            title: realTimeDataPipelines.title,
            content: realTimeDataPipelines.content,
            quiz: realTimeDataPipelinesQuiz.map(transformQuiz),
            multipleChoice: realTimeDataPipelinesMultipleChoice.map(transformMC),
        },
        {
            id: dataNormalization.id,
            title: dataNormalization.title,
            content: dataNormalization.content,
            quiz: dataNormalizationQuiz.map(transformQuiz),
            multipleChoice: dataNormalizationMultipleChoice.map(transformMC),
        },
        {
            id: timestampManagement.id,
            title: timestampManagement.title,
            content: timestampManagement.content,
            quiz: timestampManagementQuiz.map(transformQuiz),
            multipleChoice: timestampManagementMultipleChoice.map(transformMC),
        },
        {
            id: marketDataStorage.id,
            title: marketDataStorage.title,
            content: marketDataStorage.content,
            quiz: marketDataStorageQuiz.map(transformQuiz),
            multipleChoice: marketDataStorageMultipleChoice.map(transformMC),
        },
        {
            id: dataQualityValidation.id,
            title: dataQualityValidation.title,
            content: dataQualityValidation.content,
            quiz: dataQualityValidationQuiz.map(transformQuiz),
            multipleChoice: dataQualityValidationMultipleChoice.map(transformMC),
        },
        {
            id: lowLatencyProcessing.id,
            title: lowLatencyProcessing.title,
            content: lowLatencyProcessing.content,
            quiz: lowLatencyProcessingQuiz.map(transformQuiz),
            multipleChoice: lowLatencyProcessingMultipleChoice.map(transformMC),
        },
        {
            id: realTimePlatformProject.id,
            title: realTimePlatformProject.title,
            content: realTimePlatformProject.content,
            quiz: realTimePlatformProjectQuiz.map(transformQuiz),
            multipleChoice: realTimePlatformProjectMultipleChoice.map(transformMC),
        },
    ],
    quizzes: [
        ...marketDataFundamentalsQuiz.map(transformQuiz),
        ...dataFeedProtocolsQuiz.map(transformQuiz),
        ...tickDataProcessingQuiz.map(transformQuiz),
        ...ohlcvBarConstructionQuiz.map(transformQuiz),
        ...levelDataQuiz.map(transformQuiz),
        ...marketDataVendorsApisQuiz.map(transformQuiz),
        ...realTimeDataPipelinesQuiz.map(transformQuiz),
        ...dataNormalizationQuiz.map(transformQuiz),
        ...timestampManagementQuiz.map(transformQuiz),
        ...marketDataStorageQuiz.map(transformQuiz),
        ...dataQualityValidationQuiz.map(transformQuiz),
        ...lowLatencyProcessingQuiz.map(transformQuiz),
        ...realTimePlatformProjectQuiz.map(transformQuiz),
    ],
    multipleChoiceQuestions: [
        ...marketDataFundamentalsMultipleChoice.map(transformMC),
        ...dataFeedProtocolsMultipleChoice.map(transformMC),
        ...tickDataProcessingMultipleChoice.map(transformMC),
        ...ohlcvBarConstructionMultipleChoice.map(transformMC),
        ...levelDataMultipleChoice.map(transformMC),
        ...marketDataVendorsApisMultipleChoice.map(transformMC),
        ...realTimeDataPipelinesMultipleChoice.map(transformMC),
        ...dataNormalizationMultipleChoice.map(transformMC),
        ...timestampManagementMultipleChoice.map(transformMC),
        ...marketDataStorageMultipleChoice.map(transformMC),
        ...dataQualityValidationMultipleChoice.map(transformMC),
        ...lowLatencyProcessingMultipleChoice.map(transformMC),
        ...realTimePlatformProjectMultipleChoice.map(transformMC),
    ],
};
