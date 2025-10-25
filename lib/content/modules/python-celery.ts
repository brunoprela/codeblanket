/**
 * Module: Celery & Distributed Task Processing
 *
 * Complete guide to building scalable background task systems with Celery
 */

import { Module } from '../../types';

// Section imports
import { taskQueueFundamentals } from '../sections/python-celery/task-queue-fundamentals';
import { celeryArchitecture } from '../sections/python-celery/celery-architecture';
import { writingFirstTasks } from '../sections/python-celery/writing-first-tasks';
import { taskConfigurationRouting } from '../sections/python-celery/task-configuration-routing';
import { celeryBeatPeriodicTasks } from '../sections/python-celery/celery-beat-periodic-tasks';
import { taskResultsStateManagement } from '../sections/python-celery/task-results-state-management';
import { errorHandlingRetriesTimeouts } from '../sections/python-celery/error-handling-retries-timeouts';
import { monitoringWithFlower } from '../sections/python-celery/monitoring-with-flower';
import { redisVsRabbitmq } from '../sections/python-celery/redis-vs-rabbitmq';
import { distributedTaskProcessingPatterns } from '../sections/python-celery/distributed-task-processing-patterns';
import { canvasChainsGroupsChords } from '../sections/python-celery/canvas-chains-groups-chords';
import { celeryInProduction } from '../sections/python-celery/celery-in-production';
import { alternativeTaskQueues } from '../sections/python-celery/alternative-task-queues';

// Quiz imports
import { taskQueueFundamentalsQuiz } from '../quizzes/python-celery/task-queue-fundamentals';
import { celeryArchitectureQuiz } from '../quizzes/python-celery/celery-architecture';
import { writingFirstTasksQuiz } from '../quizzes/python-celery/writing-first-tasks';
import { taskConfigurationRoutingQuiz } from '../quizzes/python-celery/task-configuration-routing';
import { celeryBeatPeriodicTasksQuiz } from '../quizzes/python-celery/celery-beat-periodic-tasks';
import { taskResultsStateManagementQuiz } from '../quizzes/python-celery/task-results-state-management';
import { errorHandlingRetriesTimeoutsQuiz } from '../quizzes/python-celery/error-handling-retries-timeouts';
import { monitoringWithFlowerQuiz } from '../quizzes/python-celery/monitoring-with-flower';
import { redisVsRabbitmqQuiz } from '../quizzes/python-celery/redis-vs-rabbitmq';
import { distributedTaskProcessingPatternsQuiz } from '../quizzes/python-celery/distributed-task-processing-patterns';
import { canvasChainsGroupsChordsQuiz } from '../quizzes/python-celery/canvas-chains-groups-chords';
import { celeryInProductionQuiz } from '../quizzes/python-celery/celery-in-production';
import { alternativeTaskQueuesQuiz } from '../quizzes/python-celery/alternative-task-queues';

// Multiple choice imports
import { taskQueueFundamentalsMultipleChoice } from '../multiple-choice/python-celery/task-queue-fundamentals';
import { celeryArchitectureMultipleChoice } from '../multiple-choice/python-celery/celery-architecture';
import { writingFirstTasksMultipleChoice } from '../multiple-choice/python-celery/writing-first-tasks';
import { taskConfigurationRoutingMultipleChoice } from '../multiple-choice/python-celery/task-configuration-routing';
import { celeryBeatPeriodicTasksMultipleChoice } from '../multiple-choice/python-celery/celery-beat-periodic-tasks';
import { taskResultsStateManagementMultipleChoice } from '../multiple-choice/python-celery/task-results-state-management';
import { errorHandlingRetriesTimeoutsMultipleChoice } from '../multiple-choice/python-celery/error-handling-retries-timeouts';
import { monitoringWithFlowerMultipleChoice } from '../multiple-choice/python-celery/monitoring-with-flower';
import { redisVsRabbitmqMultipleChoice } from '../multiple-choice/python-celery/redis-vs-rabbitmq';
import { distributedTaskProcessingPatternsMultipleChoice } from '../multiple-choice/python-celery/distributed-task-processing-patterns';
import { canvasChainsGroupsChordsMultipleChoice } from '../multiple-choice/python-celery/canvas-chains-groups-chords';
import { celeryInProductionMultipleChoice } from '../multiple-choice/python-celery/celery-in-production';
import { alternativeTaskQueuesMultipleChoice } from '../multiple-choice/python-celery/alternative-task-queues';

export const pythonCeleryModule: Module = {
  id: 'python-celery',
  title: 'Celery & Distributed Task Processing',
  description:
    'Master asynchronous task processing with Celery. Build scalable background job systems for email sending, report generation, data processing, and scheduled tasks. Learn Redis vs RabbitMQ, error handling, monitoring, and production deployment.',
  icon: '🔄',
  difficulty: 'Intermediate',
  estimatedTime: '24 hours',

  sections: [
    {
      ...taskQueueFundamentals,
      quiz: taskQueueFundamentalsQuiz,
      multipleChoice: taskQueueFundamentalsMultipleChoice,
    },
    {
      ...celeryArchitecture,
      quiz: celeryArchitectureQuiz,
      multipleChoice: celeryArchitectureMultipleChoice,
    },
    {
      ...writingFirstTasks,
      quiz: writingFirstTasksQuiz,
      multipleChoice: writingFirstTasksMultipleChoice,
    },
    {
      ...taskConfigurationRouting,
      quiz: taskConfigurationRoutingQuiz,
      multipleChoice: taskConfigurationRoutingMultipleChoice,
    },
    {
      ...celeryBeatPeriodicTasks,
      quiz: celeryBeatPeriodicTasksQuiz,
      multipleChoice: celeryBeatPeriodicTasksMultipleChoice,
    },
    {
      ...taskResultsStateManagement,
      quiz: taskResultsStateManagementQuiz,
      multipleChoice: taskResultsStateManagementMultipleChoice,
    },
    {
      ...errorHandlingRetriesTimeouts,
      quiz: errorHandlingRetriesTimeoutsQuiz,
      multipleChoice: errorHandlingRetriesTimeoutsMultipleChoice,
    },
    {
      ...monitoringWithFlower,
      quiz: monitoringWithFlowerQuiz,
      multipleChoice: monitoringWithFlowerMultipleChoice,
    },
    {
      ...redisVsRabbitmq,
      quiz: redisVsRabbitmqQuiz,
      multipleChoice: redisVsRabbitmqMultipleChoice,
    },
    {
      ...distributedTaskProcessingPatterns,
      quiz: distributedTaskProcessingPatternsQuiz,
      multipleChoice: distributedTaskProcessingPatternsMultipleChoice,
    },
    {
      ...canvasChainsGroupsChords,
      quiz: canvasChainsGroupsChordsQuiz,
      multipleChoice: canvasChainsGroupsChordsMultipleChoice,
    },
    {
      ...celeryInProduction,
      quiz: celeryInProductionQuiz,
      multipleChoice: celeryInProductionMultipleChoice,
    },
    {
      ...alternativeTaskQueues,
      quiz: alternativeTaskQueuesQuiz,
      multipleChoice: alternativeTaskQueuesMultipleChoice,
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
      estimatedTime: '6 hours',
    },
    {
      title: 'Video Processing Pipeline',
      description:
        'Create a video processing pipeline: upload, transcode to multiple formats, generate thumbnails, and deliver via CDN',
      difficulty: 'Advanced',
      estimatedTime: '10 hours',
    },
    {
      title: 'Financial Report Generator',
      description:
        'Build a daily report generator using map-reduce to process millions of transactions and generate PDFs',
      difficulty: 'Advanced',
      estimatedTime: '8 hours',
    },
    {
      title: 'Social Media Scheduler',
      description:
        'Implement a social media post scheduler with Celery Beat for multiple platforms (Twitter, Facebook, LinkedIn)',
      difficulty: 'Intermediate',
      estimatedTime: '5 hours',
    },
  ],

};
