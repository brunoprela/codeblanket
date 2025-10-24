/**
 * Module: RAG & Semantic Search
 * Module 11 of Applied AI Curriculum
 */

import { Module } from '../../types';

// Section imports
import { ragFundamentals } from '../sections/rag-semantic-search/rag-fundamentals';
import { textEmbeddingsDeepDive } from '../sections/rag-semantic-search/text-embeddings-deep-dive';
import { chunkingStrategies } from '../sections/rag-semantic-search/chunking-strategies';
import { vectorDatabases } from '../sections/rag-semantic-search/vector-databases';
import { semanticSearchImplementation } from '../sections/rag-semantic-search/semantic-search-implementation';
import { advancedRetrievalStrategies } from '../sections/rag-semantic-search/advanced-retrieval-strategies';
import { reRankingRelevance } from '../sections/rag-semantic-search/re-ranking-relevance';
import { queryUnderstandingExpansion } from '../sections/rag-semantic-search/query-understanding-expansion';
import { ragEvaluation } from '../sections/rag-semantic-search/rag-evaluation';
import { conversationalRag } from '../sections/rag-semantic-search/conversational-rag';
import { multiIndexRouting } from '../sections/rag-semantic-search/multi-index-routing';
import { productionRagSystems } from '../sections/rag-semantic-search/production-rag-systems';
import { buildingKnowledgeBase } from '../sections/rag-semantic-search/building-knowledge-base';

// Quiz imports
import { ragFundamentalsQuiz } from '../quizzes/rag-semantic-search/rag-fundamentals';
import { textEmbeddingsDeepDiveQuiz } from '../quizzes/rag-semantic-search/text-embeddings-deep-dive';
import { chunkingStrategiesQuiz } from '../quizzes/rag-semantic-search/chunking-strategies';
import { vectorDatabasesQuiz } from '../quizzes/rag-semantic-search/vector-databases';
import { semanticSearchImplementationQuiz } from '../quizzes/rag-semantic-search/semantic-search-implementation';
import { advancedRetrievalStrategiesQuiz } from '../quizzes/rag-semantic-search/advanced-retrieval-strategies';
import { reRankingRelevanceQuiz } from '../quizzes/rag-semantic-search/re-ranking-relevance';
import { queryUnderstandingExpansionQuiz } from '../quizzes/rag-semantic-search/query-understanding-expansion';
import { ragEvaluationQuiz } from '../quizzes/rag-semantic-search/rag-evaluation';
import { conversationalRagQuiz } from '../quizzes/rag-semantic-search/conversational-rag';
import { multiIndexRoutingQuiz } from '../quizzes/rag-semantic-search/multi-index-routing';
import { productionRagSystemsQuiz } from '../quizzes/rag-semantic-search/production-rag-systems';
import { buildingKnowledgeBaseQuiz } from '../quizzes/rag-semantic-search/building-knowledge-base';

// Multiple choice imports
import { ragFundamentalsMC as ragFundamentalsMultipleChoice } from '../multiple-choice/rag-semantic-search/rag-fundamentals';
import { textEmbeddingsDeepDiveMC } from '../multiple-choice/rag-semantic-search/text-embeddings-deep-dive';
import { chunkingStrategiesMC } from '../multiple-choice/rag-semantic-search/chunking-strategies';
import { vectorDatabasesMC } from '../multiple-choice/rag-semantic-search/vector-databases';
import { semanticSearchImplementationMC } from '../multiple-choice/rag-semantic-search/semantic-search-implementation';
import { advancedRetrievalStrategiesMC } from '../multiple-choice/rag-semantic-search/advanced-retrieval-strategies';
import { reRankingRelevanceMC } from '../multiple-choice/rag-semantic-search/re-ranking-relevance';
import { queryUnderstandingExpansionMC } from '../multiple-choice/rag-semantic-search/query-understanding-expansion';
import { ragEvaluationMC } from '../multiple-choice/rag-semantic-search/rag-evaluation';
import { conversationalRagMC } from '../multiple-choice/rag-semantic-search/conversational-rag';
import { multiIndexRoutingMC } from '../multiple-choice/rag-semantic-search/multi-index-routing';
import { productionRagSystemsMC } from '../multiple-choice/rag-semantic-search/production-rag-systems';
import { buildingKnowledgeBaseMC } from '../multiple-choice/rag-semantic-search/building-knowledge-base';

// Helper to transform quiz format
const transformQuiz = (quiz: {
  questions: { id: number; question: string; expectedAnswer: string }[];
}) => {
  return quiz.questions.map((q) => ({
    id: q.id.toString(),
    question: q.question,
    sampleAnswer: q.expectedAnswer,
    keyPoints: [],
  }));
};

// Helper to transform multiple choice format
const transformMC = (mc: {
  questions: {
    id: number;
    question: string;
    options: string[];
    correctAnswer: number;
    explanation: string;
  }[];
}) => {
  return mc.questions.map((q) => ({
    id: q.id.toString(),
    question: q.question,
    options: q.options,
    correctAnswer: q.correctAnswer,
    explanation: q.explanation,
  }));
};

export const ragSemanticSearchModule: Module = {
  id: 'rag-semantic-search',
  title: 'RAG & Semantic Search',
  description:
    'Master Retrieval-Augmented Generation and build intelligent search systems. Learn embeddings, vector databases, advanced retrieval strategies, and production-grade RAG systems that power modern AI applications.',
  icon: '🔎',
  keyTakeaways: [
    'Master RAG fundamentals and understand why RAG is essential for production AI',
    'Deep dive into text embeddings and similarity metrics for semantic search',
    'Implement effective chunking strategies for optimal retrieval quality',
    'Compare and use vector databases: FAISS, Pinecone, Weaviate, Chroma, Qdrant',
    'Build production-grade semantic search engines with caching and monitoring',
    'Implement advanced retrieval strategies: MMR, HyDE, parent-child, multi-query',
    'Re-rank search results and implement relevance scoring algorithms',
    'Process queries with expansion, intent detection, and preprocessing',
    'Evaluate RAG systems using precision, recall, and end-to-end metrics',
    'Build conversational RAG with multi-turn context management',
    'Implement multi-index routing and query classification',
    'Deploy production RAG systems with error handling and observability',
    'Create complete knowledge base systems with document management',
  ],
  sections: [
    {
      id: 'rag-fundamentals',
      ...ragFundamentals,
      quiz: transformQuiz(ragFundamentalsQuiz),
      multipleChoice: transformMC(ragFundamentalsMultipleChoice),
    },
    {
      id: 'text-embeddings-deep-dive',
      ...textEmbeddingsDeepDive,
      quiz: transformQuiz(textEmbeddingsDeepDiveQuiz),
      multipleChoice: transformMC(textEmbeddingsDeepDiveMC),
    },
    {
      id: 'chunking-strategies',
      ...chunkingStrategies,
      quiz: transformQuiz(chunkingStrategiesQuiz),
      multipleChoice: transformMC(chunkingStrategiesMC),
    },
    {
      id: 'vector-databases',
      ...vectorDatabases,
      quiz: transformQuiz(vectorDatabasesQuiz),
      multipleChoice: transformMC(vectorDatabasesMC),
    },
    {
      id: 'semantic-search-implementation',
      ...semanticSearchImplementation,
      quiz: transformQuiz(semanticSearchImplementationQuiz),
      multipleChoice: transformMC(semanticSearchImplementationMC),
    },
    {
      id: 'advanced-retrieval-strategies',
      ...advancedRetrievalStrategies,
      quiz: transformQuiz(advancedRetrievalStrategiesQuiz),
      multipleChoice: transformMC(advancedRetrievalStrategiesMC),
    },
    {
      id: 're-ranking-relevance',
      ...reRankingRelevance,
      quiz: transformQuiz(reRankingRelevanceQuiz),
      multipleChoice: transformMC(reRankingRelevanceMC),
    },
    {
      id: 'query-understanding-expansion',
      ...queryUnderstandingExpansion,
      quiz: transformQuiz(queryUnderstandingExpansionQuiz),
      multipleChoice: transformMC(queryUnderstandingExpansionMC),
    },
    {
      id: 'rag-evaluation',
      ...ragEvaluation,
      quiz: transformQuiz(ragEvaluationQuiz),
      multipleChoice: transformMC(ragEvaluationMC),
    },
    {
      id: 'conversational-rag',
      ...conversationalRag,
      quiz: transformQuiz(conversationalRagQuiz),
      multipleChoice: transformMC(conversationalRagMC),
    },
    {
      id: 'multi-index-routing',
      ...multiIndexRouting,
      quiz: transformQuiz(multiIndexRoutingQuiz),
      multipleChoice: transformMC(multiIndexRoutingMC),
    },
    {
      id: 'production-rag-systems',
      ...productionRagSystems,
      quiz: transformQuiz(productionRagSystemsQuiz),
      multipleChoice: transformMC(productionRagSystemsMC),
    },
    {
      id: 'building-knowledge-base',
      ...buildingKnowledgeBase,
      quiz: transformQuiz(buildingKnowledgeBaseQuiz),
      multipleChoice: transformMC(buildingKnowledgeBaseMC),
    },
  ],
};
