import { Quiz } from '@/lib/types';

export const moduleProjectMultipleChoice: Quiz = {
  title: 'Module Project Quiz',
  description:
    'Test your understanding of the Financial Data Aggregator project requirements and design.',
  questions: [
    {
      id: 'project-1',
      question:
        'In the Financial Data Aggregator project, why is it important to implement retry logic with exponential backoff when fetching data from APIs?',
      options: [
        'To make the code more complex and impressive',
        'To handle temporary failures (network issues, API rate limits) gracefully without losing data, increasing reliability',
        'To slow down the data collection process',
        'It is not important; APIs always work perfectly',
      ],
      correctAnswer: 1,
      explanation:
        'Exponential backoff retry logic is critical for production systems. APIs fail temporarily due to: network blips, rate limit exceeded, server maintenance. Without retries, you lose data. Exponential backoff (wait 2s, then 4s, then 8s) prevents overwhelming a struggling API while giving it time to recover. The project uses tenacity library: @retry(stop=stop_after_attempt(3), wait=wait_exponential(...)). Real-world impact: collection script runs overnight, encounters temporary failure at 3 AM, automatically retries and succeeds, vs failing and requiring manual re-run.',
    },
    {
      id: 'project-2',
      question:
        'Why does the project use TimescaleDB hypertables instead of regular PostgreSQL tables for storing price data?',
      options: [
        'Hypertables are required by law for financial data',
        'Hypertables automatically partition time-series data by time, providing 10-100x query performance improvement for date-range queries',
        'Hypertables use less disk space than regular tables',
        'Hypertables are easier to set up than regular tables',
      ],
      correctAnswer: 1,
      explanation:
        'TimescaleDB hypertables automatically partition data into time-based chunks (e.g., weekly). Query for Jan 2024 data? Only Jan 2024 chunk is scanned. Regular PostgreSQL table would scan entire multi-year dataset. Performance: query 1 year of data from 10 years total - hypertable scans 1/10 of data (fast), regular table scans everything (slow). Additional benefits: compression (90%+ space savings), continuous aggregates (pre-computed OHLCV bars), retention policies (auto-delete old data). Essential for multi-year tick/minute data. Setup cost: just SELECT create_hypertable() after CREATE TABLE.',
    },
    {
      id: 'project-3',
      question:
        'In the project architecture, what is the purpose of the CollectorManager class?',
      options: [
        'To display data on a dashboard',
        'To provide fallback between data sources and coordinate fetching from multiple APIs with consistent interface',
        'To store data in the database',
        'To generate API keys',
      ],
      correctAnswer: 1,
      explanation:
        'CollectorManager abstracts multiple data sources behind unified interface. Benefits: (1) Try primary source (Yahoo Finance), if fails automatically try fallback (Alpha Vantage), (2) Consistent API regardless of source, (3) Easy to add new sources, (4) Parallel fetching coordination. Real-world scenario: Yahoo Finance has outage → CollectorManager silently switches to Alpha Vantage → data collection continues uninterrupted. Alternative design (bad): hardcode yahoo collector everywhere, when Yahoo fails everything breaks. Good design pattern for any multi-source system.',
    },
    {
      id: 'project-4',
      question:
        'Why does the project validate data (check OHLC relationships, null values) before inserting into database?',
      options: [
        'Database validation is too slow, so we validate in Python',
        'To catch data quality issues early (before they corrupt analysis), prevent invalid data from entering database, and fail fast with clear errors',
        'Validation is not necessary if data comes from reliable sources',
        'To make the code longer and more professional-looking',
      ],
      correctAnswer: 1,
      explanation:
        'Early validation prevents downstream disasters. Scenarios: (1) API returns corrupted data (High < Low), validator catches it, logs error, skips insert vs. bad data enters database, analysis produces wrong signals, strategy loses money, (2) API returns nulls, validator catches vs. database query fails weeks later in production, (3) Negative prices/volumes, validator rejects vs. breaks calculations. Best practice: validate at boundary (data entry point). Database constraints are backup but slower and less informative errors. Validation should check: nulls, OHLC relationships, positive prices/volumes, reasonable ranges, duplicates.',
    },
    {
      id: 'project-5',
      question:
        'What is the main advantage of using Docker Compose for the project instead of installing PostgreSQL/Redis directly on your system?',
      options: [
        'Docker makes databases run faster',
        'Docker Compose provides reproducible environment, easy setup, isolation from system, and simple cleanup - ensuring project works identically for all users',
        'Docker Compose is required for all Python projects',
        'Docker reduces electricity usage',
      ],
      correctAnswer: 1,
      explanation:
        'Docker Compose solves "works on my machine" for dependencies. Benefits: (1) New team member: git clone, docker-compose up, everything works in 5 minutes vs. install PostgreSQL, configure, troubleshoot version conflicts (hours), (2) Isolation: project PostgreSQL on port 5432, other project on 5433, no conflicts, (3) Clean environment: docker-compose down removes everything, no leftover services, (4) Version control: docker-compose.yml specifies exact versions (postgres:15, redis:latest), (5) Production parity: same Docker containers in dev and prod. One-time Docker setup cost, massive ongoing productivity gain.',
    },
  ],
};
