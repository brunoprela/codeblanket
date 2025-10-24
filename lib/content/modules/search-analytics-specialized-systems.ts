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
      id: 'full-text-search-fundamentals',
      title: 'Full-Text Search Fundamentals',
      topics: [
        'Inverted indexes',
        'Tokenization and analysis',
        'TF-IDF scoring',
        'Relevance ranking',
        'Fuzzy matching',
        'Search quality metrics',
      ],
    },
    {
      id: 'elasticsearch-architecture',
      title: 'Elasticsearch Architecture',
      topics: [
        'Cluster, nodes, shards, replicas',
        'Document indexing',
        'Mapping and data types',
        'Query DSL',
        'Aggregations',
        'Scaling Elasticsearch',
      ],
    },
    {
      id: 'search-optimization',
      title: 'Search Optimization',
      topics: [
        'Index design',
        'Query performance',
        'Caching strategies',
        'Shard sizing',
        'Relevance tuning',
        'Autocomplete and suggestions',
      ],
    },
    {
      id: 'analytics-data-pipeline',
      title: 'Analytics Data Pipeline',
      topics: [
        'Data ingestion',
        'ETL vs ELT',
        'Data lake vs data warehouse',
        'Lambda architecture',
        'Kappa architecture',
        'Real-time vs batch analytics',
      ],
    },
    {
      id: 'column-oriented-databases',
      title: 'Column-Oriented Databases',
      topics: [
        'Columnar storage benefits',
        'Use cases: Analytics, OLAP',
        'Compression techniques',
        'ClickHouse, Druid, BigQuery',
        'Query performance',
        'When to use columnar stores',
      ],
    },
    {
      id: 'data-warehousing',
      title: 'Data Warehousing',
      topics: [
        'Star schema',
        'Snowflake schema',
        'Fact tables and dimension tables',
        'Slowly changing dimensions',
        'Redshift, Snowflake, BigQuery',
        'MPP (Massively Parallel Processing)',
      ],
    },
    {
      id: 'real-time-analytics',
      title: 'Real-Time Analytics',
      topics: [
        'Stream processing for analytics',
        'Approximation algorithms',
        'Real-time dashboards',
        'Druid, Pinot',
        'Trade-offs: Latency vs accuracy',
      ],
    },
    {
      id: 'log-analytics',
      title: 'Log Analytics',
      topics: [
        'ELK Stack',
        'Log aggregation pipeline',
        'Log parsing and structuring',
        'Visualization',
        'Alerting on logs',
        'Cost optimization',
      ],
    },
    {
      id: 'time-series-databases',
      title: 'Time-Series Databases',
      topics: [
        'Time-series data characteristics',
        'InfluxDB, TimescaleDB, Prometheus',
        'Downsampling and retention policies',
        'Querying time-series data',
        'Use cases: Metrics, IoT',
        'Storage optimization',
      ],
    },
    {
      id: 'graph-databases',
      title: 'Graph Databases',
      topics: [
        'Graph data model',
        'Graph traversal algorithms',
        'Neo4j, Amazon Neptune',
        'Use cases and applications',
        'Graph query languages',
        'When to use graph databases',
      ],
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
  content: {
    sections: {
      'full-text-search-fundamentals': fullTextSearchFundamentalsSection,
      'elasticsearch-architecture': elasticsearchArchitectureSection,
      'search-optimization': searchOptimizationSection,
      'analytics-data-pipeline': analyticsDataPipelineSection,
      'column-oriented-databases': columnOrientedDatabasesSection,
      'data-warehousing': dataWarehousingSection,
      'real-time-analytics': realTimeAnalyticsSection,
      'log-analytics': logAnalyticsSection,
      'time-series-databases': timeSeriesDatabasesSection,
      'graph-databases': graphDatabasesSection,
    },
    quizzes: {
      'full-text-search-fundamentals-discussion':
        fullTextSearchFundamentalsDiscussionQuiz,
      'elasticsearch-architecture-discussion':
        elasticsearchArchitectureDiscussionQuiz,
      'search-optimization-discussion': searchOptimizationDiscussionQuiz,
      'analytics-data-pipeline-discussion': analyticsDataPipelineDiscussionQuiz,
      'column-oriented-databases-discussion':
        columnOrientedDatabasesDiscussionQuiz,
      'data-warehousing-discussion': dataWarehousingDiscussionQuiz,
      'real-time-analytics-discussion': realTimeAnalyticsDiscussionQuiz,
      'log-analytics-discussion': logAnalyticsDiscussionQuiz,
      'time-series-databases-discussion': timeSeriesDatabasesDiscussionQuiz,
      'graph-databases-discussion': graphDatabasesDiscussionQuiz,
    },
    multipleChoice: {
      'full-text-search-fundamentals-mcq': fullTextSearchFundamentalsMCQ,
      'elasticsearch-architecture-mcq': elasticsearchArchitectureMCQ,
      'search-optimization-mcq': searchOptimizationMCQ,
      'analytics-data-pipeline-mcq': analyticsDataPipelineMCQ,
      'column-oriented-databases-mcq': columnOrientedDatabasesMCQ,
      'data-warehousing-mcq': dataWarehousingMCQ,
      'real-time-analytics-mcq': realTimeAnalyticsMCQ,
      'log-analytics-mcq': logAnalyticsMCQ,
      'time-series-databases-mcq': timeSeriesDatabasesMCQ,
      'graph-databases-mcq': graphDatabasesMCQ,
    },
  },
};

export default searchAnalyticsSpecializedSystemsModule;
