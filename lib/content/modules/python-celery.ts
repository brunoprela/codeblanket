/**
 * Module: Celery & Distributed Task Processing
 *
 * Complete guide to building scalable background task systems with Celery
 */

import { Module } from '../types';

// Section imports
import { taskQueueFundamentals } from './sections/python-celery/task-queue-fundamentals';
import { celeryArchitecture } from './sections/python-celery/celery-architecture';
import { writingFirstTasks } from './sections/python-celery/writing-first-tasks';
import { taskConfigurationRouting } from './sections/python-celery/task-configuration-routing';
import { celeryBeatPeriodicTasks } from './sections/python-celery/celery-beat-periodic-tasks';
import { taskResultsStateManagement } from './sections/python-celery/task-results-state-management';
import { errorHandlingRetriesTimeouts } from './sections/python-celery/error-handling-retries-timeouts';
import { monitoringWithFlower } from './sections/python-celery/monitoring-with-flower';
import { redisVsRabbitmq } from './sections/python-celery/redis-vs-rabbitmq';
import { distributedTaskProcessingPatterns } from './sections/python-celery/distributed-task-processing-patterns';
import { canvasChainsGroupsChords } from './sections/python-celery/canvas-chains-groups-chords';
import { celeryInProduction } from './sections/python-celery/celery-in-production';
import { alternativeTaskQueues } from './sections/python-celery/alternative-task-queues';

// Quiz imports
import { taskQueueFundamentalsQuiz } from './quizzes/python-celery/task-queue-fundamentals';
import { celeryArchitectureQuiz } from './quizzes/python-celery/celery-architecture';
import { writingFirstTasksQuiz } from './quizzes/python-celery/writing-first-tasks';
import { taskConfigurationRoutingQuiz } from './quizzes/python-celery/task-configuration-routing';
import { celeryBeatPeriodicTasksQuiz } from './quizzes/python-celery/celery-beat-periodic-tasks';
import { taskResultsStateManagementQuiz } from './quizzes/python-celery/task-results-state-management';
import { errorHandlingRetriesTimeoutsQuiz } from './quizzes/python-celery/error-handling-retries-timeouts';
import { monitoringWithFlowerQuiz } from './quizzes/python-celery/monitoring-with-flower';
import { redisVsRabbitmqQuiz } from './quizzes/python-celery/redis-vs-rabbitmq';
import { distributedTaskProcessingPatternsQuiz } from './quizzes/python-celery/distributed-task-processing-patterns';
import { canvasChainsGroupsChordsQuiz } from './quizzes/python-celery/canvas-chains-groups-chords';
import { celeryInProductionQuiz } from './quizzes/python-celery/celery-in-production';
import { alternativeTaskQueuesQuiz } from './quizzes/python-celery/alternative-task-queues';

// Multiple choice imports
import { taskQueueFundamentalsMultipleChoice } from './multiple-choice/python-celery/task-queue-fundamentals';
import { celeryArchitectureMultipleChoice } from './multiple-choice/python-celery/celery-architecture';
import { writingFirstTasksMultipleChoice } from './multiple-choice/python-celery/writing-first-tasks';
import { taskConfigurationRoutingMultipleChoice } from './multiple-choice/python-celery/task-configuration-routing';
import { celeryBeatPeriodicTasksMultipleChoice } from './multiple-choice/python-celery/celery-beat-periodic-tasks';
import { taskResultsStateManagementMultipleChoice } from './multiple-choice/python-celery/task-results-state-management';
import { errorHandlingRetriesTimeoutsMultipleChoice } from './multiple-choice/python-celery/error-handling-retries-timeouts';
import { monitoringWithFlowerMultipleChoice } from './multiple-choice/python-celery/monitoring-with-flower';
import { redisVsRabbitmqMultipleChoice } from './multiple-choice/python-celery/redis-vs-rabbitmq';
import { distributedTaskProcessingPatternsMultipleChoice } from './multiple-choice/python-celery/distributed-task-processing-patterns';
import { canvasChainsGroupsChordsMultipleChoice } from './multiple-choice/python-celery/canvas-chains-groups-chords';
import { celeryInProductionMultipleChoice } from './multiple-choice/python-celery/celery-in-production';
import { alternativeTaskQueuesMultipleChoice } from './multiple-choice/python-celery/alternative-task-queues';

export const pythonCeleryModule: Module = {
  id: 'python-celery',
  title: 'Celery & Distributed Task Processing',
  description:
    'Master asynchronous task processing with Celery. Build scalable background job systems for email sending, report generation, data processing, and scheduled tasks. Learn Redis vs RabbitMQ, error handling, monitoring, and production deployment.',
  icon: '🔄',
  difficulty: 'Intermediate',
  estimatedHours: 24,
  topic: 'Python',
  curriculum: 'python',

  sections: [
    {
      id: 'task-queue-fundamentals',
      title: 'Task Queue Fundamentals',
      content: taskQueueFundamentals.content,
      quiz: taskQueueFundamentalsQuiz,
      multipleChoice: taskQueueFundamentalsMultipleChoice,
      order: 1,
      estimatedMinutes: 90,
    },
    {
      id: 'celery-architecture',
      title: 'Celery Architecture & Components',
      content: celeryArchitecture.content,
      quiz: celeryArchitectureQuiz,
      multipleChoice: celeryArchitectureMultipleChoice,
      order: 2,
      estimatedMinutes: 100,
    },
    {
      id: 'writing-first-tasks',
      title: 'Writing Your First Celery Tasks',
      content: writingFirstTasks.content,
      quiz: writingFirstTasksQuiz,
      multipleChoice: writingFirstTasksMultipleChoice,
      order: 3,
      estimatedMinutes: 120,
    },
    {
      id: 'task-configuration-routing',
      title: 'Task Configuration & Routing',
      content: taskConfigurationRouting.content,
      quiz: taskConfigurationRoutingQuiz,
      multipleChoice: taskConfigurationRoutingMultipleChoice,
      order: 4,
      estimatedMinutes: 110,
    },
    {
      id: 'celery-beat-periodic-tasks',
      title: 'Celery Beat & Periodic Tasks',
      content: celeryBeatPeriodicTasks.content,
      quiz: celeryBeatPeriodicTasksQuiz,
      multipleChoice: celeryBeatPeriodicTasksMultipleChoice,
      order: 5,
      estimatedMinutes: 100,
    },
    {
      id: 'task-results-state-management',
      title: 'Task Results & State Management',
      content: taskResultsStateManagement.content,
      quiz: taskResultsStateManagementQuiz,
      multipleChoice: taskResultsStateManagementMultipleChoice,
      order: 6,
      estimatedMinutes: 90,
    },
    {
      id: 'error-handling-retries-timeouts',
      title: 'Error Handling, Retries & Timeouts',
      content: errorHandlingRetriesTimeouts.content,
      quiz: errorHandlingRetriesTimeoutsQuiz,
      multipleChoice: errorHandlingRetriesTimeoutsMultipleChoice,
      order: 7,
      estimatedMinutes: 110,
    },
    {
      id: 'monitoring-with-flower',
      title: 'Task Monitoring with Flower',
      content: monitoringWithFlower.content,
      quiz: monitoringWithFlowerQuiz,
      multipleChoice: monitoringWithFlowerMultipleChoice,
      order: 8,
      estimatedMinutes: 90,
    },
    {
      id: 'redis-vs-rabbitmq',
      title: 'Redis vs RabbitMQ as Message Broker',
      content: redisVsRabbitmq.content,
      quiz: redisVsRabbitmqQuiz,
      multipleChoice: redisVsRabbitmqMultipleChoice,
      order: 9,
      estimatedMinutes: 100,
    },
    {
      id: 'distributed-task-processing-patterns',
      title: 'Distributed Task Processing Patterns',
      content: distributedTaskProcessingPatterns.content,
      quiz: distributedTaskProcessingPatternsQuiz,
      multipleChoice: distributedTaskProcessingPatternsMultipleChoice,
      order: 10,
      estimatedMinutes: 120,
    },
    {
      id: 'canvas-chains-groups-chords',
      title: 'Canvas: Chains, Groups, Chords',
      content: canvasChainsGroupsChords.content,
      quiz: canvasChainsGroupsChordsQuiz,
      multipleChoice: canvasChainsGroupsChordsMultipleChoice,
      order: 11,
      estimatedMinutes: 110,
    },
    {
      id: 'celery-in-production',
      title: 'Celery in Production',
      content: celeryInProduction.content,
      quiz: celeryInProductionQuiz,
      multipleChoice: celeryInProductionMultipleChoice,
      order: 12,
      estimatedMinutes: 130,
    },
    {
      id: 'alternative-task-queues',
      title: 'Alternative Task Queues (RQ, Dramatiq, Huey)',
      content: alternativeTaskQueues.content,
      quiz: alternativeTaskQueuesQuiz,
      multipleChoice: alternativeTaskQueuesMultipleChoice,
      order: 13,
      estimatedMinutes: 80,
    },
  ],

  keyTakeaways: [
    'Understand task queues and async processing patterns for scalable applications',
    'Master Celery architecture: brokers, workers, result backends, and beat scheduler',
    'Implement robust error handling with automatic retries and exponential backoff',
    'Monitor production systems with Flower and Prometheus metrics',
    'Choose the right broker (Redis vs RabbitMQ) based on reliability requirements',
    'Apply distributed patterns: map-reduce, fan-out/fan-in, task chunking',
    'Compose complex workflows with Canvas (chains, groups, chords)',
    'Deploy Celery to production with Docker, Kubernetes, and auto-scaling',
  ],

  learningObjectives: [
    'Understand when and why to use task queues vs synchronous processing',
    'Install and configure Celery with Redis and RabbitMQ brokers',
    'Write tasks with proper configuration: retries, timeouts, routing',
    'Schedule periodic tasks with Celery Beat and crontab schedules',
    'Track task state and retrieve results from result backends',
    'Implement comprehensive error handling and dead letter queues',
    'Monitor tasks in real-time with Flower dashboard and API',
    'Compare Redis vs RabbitMQ trade-offs for different use cases',
    'Apply distributed processing patterns for massive scale',
    'Orchestrate complex workflows with Canvas primitives',
    'Deploy production-ready Celery with monitoring and auto-scaling',
    'Evaluate alternative task queues (RQ, Dramatiq, Huey)',
  ],

  prerequisites: [
    'Python fundamentals (functions, classes, decorators)',
    'Understanding of synchronous vs asynchronous execution',
    'Basic knowledge of Redis or message queues (helpful)',
    'Familiarity with command line and virtual environments',
    'Docker basics (for production deployment section)',
  ],

  practicalProjects: [
    {
      title: 'Email Campaign System',
      description:
        'Build a system to send 1M marketing emails using Celery with rate limiting, retry logic, and progress tracking',
      difficulty: 'Intermediate',
      estimatedHours: 6,
    },
    {
      title: 'Video Processing Pipeline',
      description:
        'Create a video processing pipeline: upload, transcode to multiple formats, generate thumbnails, and deliver via CDN',
      difficulty: 'Advanced',
      estimatedHours: 10,
    },
    {
      title: 'Financial Report Generator',
      description:
        'Build a daily report generator using map-reduce to process millions of transactions and generate PDFs',
      difficulty: 'Advanced',
      estimatedHours: 8,
    },
    {
      title: 'Social Media Scheduler',
      description:
        'Implement a social media post scheduler with Celery Beat for multiple platforms (Twitter, Facebook, LinkedIn)',
      difficulty: 'Intermediate',
      estimatedHours: 5,
    },
  ],

  resources: [
    {
      title: 'Official Celery Documentation',
      url: 'https://docs.celeryq.dev',
      type: 'documentation',
    },
    {
      title: 'Flower Documentation',
      url: 'https://flower.readthedocs.io',
      type: 'documentation',
    },
    {
      title: 'Redis Documentation',
      url: 'https://redis.io/docs',
      type: 'documentation',
    },
    {
      title: 'RabbitMQ Tutorials',
      url: 'https://www.rabbitmq.com/getstarted.html',
      type: 'tutorial',
    },
  ],

  tags: [
    'celery',
    'task-queue',
    'async',
    'background-jobs',
    'redis',
    'rabbitmq',
    'distributed-systems',
    'scalability',
    'monitoring',
    'production',
    'workflow',
    'scheduling',
  ],
};
