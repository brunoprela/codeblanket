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
      ...ragFundamentals,
      quiz: ragFundamentalsQuiz,
      multipleChoice: ragFundamentalsMultipleChoice,
    },
    {
      ...textEmbeddingsDeepDive,
      quiz: textEmbeddingsDeepDiveQuiz,
      multipleChoice: textEmbeddingsDeepDiveMC,
    },
    {
      ...chunkingStrategies,
      quiz: chunkingStrategiesQuiz,
      multipleChoice: chunkingStrategiesMC,
    },
    {
      ...vectorDatabases,
      quiz: vectorDatabasesQuiz,
      multipleChoice: vectorDatabasesMC,
    },
    {
      ...semanticSearchImplementation,
      quiz: semanticSearchImplementationQuiz,
      multipleChoice: semanticSearchImplementationMC,
    },
    {
      ...advancedRetrievalStrategies,
      quiz: advancedRetrievalStrategiesQuiz,
      multipleChoice: advancedRetrievalStrategiesMC,
    },
    {
      ...reRankingRelevance,
      quiz: reRankingRelevanceQuiz,
      multipleChoice: reRankingRelevanceMC,
    },
    {
      ...queryUnderstandingExpansion,
      quiz: queryUnderstandingExpansionQuiz,
      multipleChoice: queryUnderstandingExpansionMC,
    },
    {
      ...ragEvaluation,
      quiz: ragEvaluationQuiz,
      multipleChoice: ragEvaluationMC,
    },
    {
      ...conversationalRag,
      quiz: conversationalRagQuiz,
      multipleChoice: conversationalRagMC,
    },
    {
      ...multiIndexRouting,
      quiz: multiIndexRoutingQuiz,
      multipleChoice: multiIndexRoutingMC,
    },
    {
      ...productionRagSystems,
      quiz: productionRagSystemsQuiz,
      multipleChoice: productionRagSystemsMC,
    },
    {
      ...buildingKnowledgeBase,
      quiz: buildingKnowledgeBaseQuiz,
      multipleChoice: buildingKnowledgeBaseMC,
    },
  ],
};
