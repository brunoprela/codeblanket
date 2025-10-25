/**
 * Multiple choice questions for Airbnb Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const airbnbarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Which search technology does Airbnb use for its listing search with complex filters?',
    options: [
      'MySQL with full-text search indexes',
      'Apache Solr with geographic queries',
      'Elasticsearch with custom ranking',
      'Amazon CloudSearch',
    ],
    correctAnswer: 2,
    explanation:
      'Airbnb uses Elasticsearch for listing search with complex filters (location, price, amenities, availability dates). Elasticsearch supports geohash-based location queries, range filters, and custom ranking models. Search results are ranked using ML models trained on booking conversions, considering base price, reviews, host quality, instant book, and user preferences. The system is sharded by geographic regions with 100+ nodes handling millions of searches.',
  },
  {
    id: 'mc2',
    question:
      'How does Airbnb handle calendar availability checking for search queries?',
    options: [
      'Real-time queries to listing databases',
      'Precomputed daily in batch jobs and indexed in Elasticsearch',
      'Cached in Redis with 1-hour TTL',
      'Client-side filtering after fetching all listings',
    ],
    correctAnswer: 1,
    explanation:
      'Airbnb precomputes calendar availability daily using batch jobs (likely Airflow orchestrating Spark). The availability data is indexed in Elasticsearch alongside listing data. This avoids real-time database queries for every search (which would be slow and expensive). When users search for specific dates, Elasticsearch filters to only available listings. This enables sub-200ms query latency for 95% of searches despite 7M+ listings.',
  },
  {
    id: 'mc3',
    question: "What is Airbnb's service fee split between guests and hosts?",
    options: [
      'Guests: 5%, Hosts: 5%',
      'Guests: 10%, Hosts: 10%',
      'Guests: 14%, Hosts: 3%',
      'Guests: 20%, Hosts: 0%',
    ],
    correctAnswer: 2,
    explanation:
      "Airbnb charges guests a 14% service fee and hosts a 3% service fee (percentages may vary by region/promotion). For a $1500 booking, the guest pays ~$210 service fee ($1710 total), and the host receives $1455 after Airbnb's $45 fee. This fee structure covers payment processing, customer support, insurance, and platform operations. Payments are held in escrow with a 24-hour cancellation window before releasing to hosts.",
  },
  {
    id: 'mc4',
    question: 'How many currencies does Airbnb support for payments?',
    options: [
      'Approximately 50 currencies',
      'Approximately 100 currencies',
      'Approximately 191 currencies',
      'All official world currencies (180+)',
    ],
    correctAnswer: 2,
    explanation:
      "Airbnb supports 191 currencies for payments. The payment system handles multi-currency conversions when guests and hosts use different currencies. For example, a US guest booking a French listing might pay in USD while the host receives EUR. Airbnb's payment service orchestrates currency conversion, fee calculations, tax withholdings, and transfers to various payout methods (bank transfer, PayPal, Payoneer) across different jurisdictions.",
  },
  {
    id: 'mc5',
    question:
      'Which workflow orchestration tool does Airbnb use for batch data processing?',
    options: [
      'Apache Airflow',
      'AWS Step Functions',
      'Luigi',
      'Kubernetes CronJobs',
    ],
    correctAnswer: 0,
    explanation:
      'Airbnb uses Apache Airflow for orchestrating batch data processing workflows. Airflow manages thousands of DAGs (Directed Acyclic Graphs) for daily jobs like: syncing database snapshots to data lake, transforming data with Spark, loading into Hive warehouse, updating search indexes, generating analytics dashboards, and training ML models. Airbnb was an early adopter and contributor to Airflow, which became an Apache project.',
  },
];
