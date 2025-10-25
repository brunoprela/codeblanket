import { Module } from '@/lib/types';

// Import sections
import fullTextSearchFundamentalsSection from '../sections/search-analytics-specialized-systems/full-text-search-fundamentals';
import elasticsearchArchitectureSection from '../sections/search-analytics-specialized-systems/elasticsearch-architecture';
import searchOptimizationSection from '../sections/search-analytics-specialized-systems/search-optimization';
import analyticsDataPipelineSection from '../sections/search-analytics-specialized-systems/analytics-data-pipeline';
import columnOrientedDatabasesSection from '../sections/search-analytics-specialized-systems/column-oriented-databases';
import dataWarehousingSection from '../sections/search-analytics-specialized-systems/data-warehousing';
import realTimeAnalyticsSection from '../sections/search-analytics-specialized-systems/real-time-analytics';
import logAnalyticsSection from '../sections/search-analytics-specialized-systems/log-analytics';
import timeSeriesDatabasesSection from '../sections/search-analytics-specialized-systems/time-series-databases';
import graphDatabasesSection from '../sections/search-analytics-specialized-systems/graph-databases';

// Import discussion quizzes
import fullTextSearchFundamentalsDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/full-text-search-fundamentals-discussion';
import elasticsearchArchitectureDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/elasticsearch-architecture-discussion';
import searchOptimizationDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/search-optimization-discussion';
import analyticsDataPipelineDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/analytics-data-pipeline-discussion';
import columnOrientedDatabasesDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/column-oriented-databases-discussion';
import dataWarehousingDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/data-warehousing-discussion';
import realTimeAnalyticsDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/real-time-analytics-discussion';
import logAnalyticsDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/log-analytics-discussion';
import timeSeriesDatabasesDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/time-series-databases-discussion';
import graphDatabasesDiscussionQuiz from '../quizzes/search-analytics-specialized-systems/graph-databases-discussion';

// Import multiple choice quizzes
import fullTextSearchFundamentalsMCQ from '../multiple-choice/search-analytics-specialized-systems/full-text-search-fundamentals-mcq';
import elasticsearchArchitectureMCQ from '../multiple-choice/search-analytics-specialized-systems/elasticsearch-architecture-mcq';
import searchOptimizationMCQ from '../multiple-choice/search-analytics-specialized-systems/search-optimization-mcq';
import analyticsDataPipelineMCQ from '../multiple-choice/search-analytics-specialized-systems/analytics-data-pipeline-mcq';
import columnOrientedDatabasesMCQ from '../multiple-choice/search-analytics-specialized-systems/column-oriented-databases-mcq';
import dataWarehousingMCQ from '../multiple-choice/search-analytics-specialized-systems/data-warehousing-mcq';
import realTimeAnalyticsMCQ from '../multiple-choice/search-analytics-specialized-systems/real-time-analytics-mcq';
import logAnalyticsMCQ from '../multiple-choice/search-analytics-specialized-systems/log-analytics-mcq';
import timeSeriesDatabasesMCQ from '../multiple-choice/search-analytics-specialized-systems/time-series-databases-mcq';
import graphDatabasesMCQ from '../multiple-choice/search-analytics-specialized-systems/graph-databases-mcq';

const searchAnalyticsSpecializedSystemsModule: Module = {
  id: 'search-analytics-specialized-systems',
  title: 'Search, Analytics & Specialized Systems',
  description:
    'Master search engines, analytics platforms, and specialized storage systems for building scalable data-intensive applications',
  icon: '🔍',
  sections: [
    {
      ...fullTextSearchFundamentalsSection,
      quiz: fullTextSearchFundamentalsDiscussionQuiz,
      multipleChoice: fullTextSearchFundamentalsMCQ,
    },
    {
      ...elasticsearchArchitectureSection,
      quiz: elasticsearchArchitectureDiscussionQuiz,
      multipleChoice: elasticsearchArchitectureMCQ,
    },
    {
      ...searchOptimizationSection,
      quiz: searchOptimizationDiscussionQuiz,
      multipleChoice: searchOptimizationMCQ,
    },
    {
      ...analyticsDataPipelineSection,
      quiz: analyticsDataPipelineDiscussionQuiz,
      multipleChoice: analyticsDataPipelineMCQ,
    },
    {
      ...columnOrientedDatabasesSection,
      quiz: columnOrientedDatabasesDiscussionQuiz,
      multipleChoice: columnOrientedDatabasesMCQ,
    },
    {
      ...dataWarehousingSection,
      quiz: dataWarehousingDiscussionQuiz,
      multipleChoice: dataWarehousingMCQ,
    },
    {
      ...realTimeAnalyticsSection,
      quiz: realTimeAnalyticsDiscussionQuiz,
      multipleChoice: realTimeAnalyticsMCQ,
    },
    {
      ...logAnalyticsSection,
      quiz: logAnalyticsDiscussionQuiz,
      multipleChoice: logAnalyticsMCQ,
    },
    {
      ...timeSeriesDatabasesSection,
      quiz: timeSeriesDatabasesDiscussionQuiz,
      multipleChoice: timeSeriesDatabasesMCQ,
    },
    {
      ...graphDatabasesSection,
      quiz: graphDatabasesDiscussionQuiz,
      multipleChoice: graphDatabasesMCQ,
    },
  ],
  keyTakeaways: [
    'Search engines use inverted indexes for efficient full-text search',
    'Elasticsearch provides distributed search with horizontal scaling',
    'Column-oriented databases excel at analytical workloads',
    'Data warehouses use dimensional modeling (star/snowflake schemas)',
    'Real-time analytics trade accuracy for low latency',
    'Time-series databases optimize for temporal data patterns',
    'Graph databases efficiently model and query connected data',
    'Analytics pipelines require careful consideration of ETL vs ELT',
  ],
  learningObjectives: [
    'Understand how full-text search engines work internally',
    'Design and optimize Elasticsearch clusters for scale',
    'Choose the right database for analytics workloads',
    'Build data pipelines for batch and real-time analytics',
    'Implement efficient log aggregation and analysis',
    'Model and query graph data effectively',
    'Apply specialized databases to domain-specific problems',
  ],
};

export default searchAnalyticsSpecializedSystemsModule;
