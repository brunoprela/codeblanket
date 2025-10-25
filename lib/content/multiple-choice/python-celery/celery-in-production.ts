/**
 * Multiple choice questions for Celery in Production section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const celeryInProductionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'How should you deploy Celery workers in production?',
    options: [
      'Run manually in terminal',
      'Use systemd/supervisor/Docker/Kubernetes with auto-restart',
      'Single worker is enough',
      'No deployment needed',
    ],
    correctAnswer: 1,
    explanation:
      'Production deployment: Use process supervisor (systemd, supervisor, Docker, K8s) with auto-restart. Run multiple workers (3+) for redundancy. systemd example: ExecStart=/usr/bin/celery worker Restart=always. Never run manually - workers must restart automatically on failure.',
  },
  {
    id: 'mc2',
    question: 'What is the minimum number of workers for production?',
    options: [
      '1 worker is enough',
      '3+ workers for redundancy',
      '100 workers minimum',
      'Workers not needed',
    ],
    correctAnswer: 1,
    explanation:
      'Minimum 3 workers for redundancy (N+2). If 1 worker fails, others continue processing. Load distribution across workers. Auto-scaling: Start with 3, scale to 20+ based on queue depth. Single worker = single point of failure (not production-ready).',
  },
  {
    id: 'mc3',
    question: 'How do you auto-scale Celery workers based on queue depth?',
    options: [
      'Manually add workers',
      'Use Kubernetes HorizontalPodAutoscaler with celery_queue_depth metric',
      'Auto-scaling not possible',
      'Use only vertical scaling',
    ],
    correctAnswer: 1,
    explanation:
      'Auto-scaling with K8s HPA: Monitor celery_queue_depth metric via Prometheus. HPA config: minReplicas: 3, maxReplicas: 20, targetAverageValue: 1000. Scales workers based on queue size. Example: 10K queue → 10 workers, 1K queue → 3 workers. Cost-effective scaling.',
  },
  {
    id: 'mc4',
    question: 'What monitoring should you implement for production Celery?',
    options: [
      'No monitoring needed',
      'Flower + Prometheus + Grafana + Alerts',
      'Just check logs occasionally',
      'Only monitor broker',
    ],
    correctAnswer: 1,
    explanation:
      'Production monitoring: (1) Flower - real-time dashboard, (2) Prometheus - metrics collection, (3) Grafana - visualization, (4) AlertManager - alerts. Monitor: queue depth, worker health, task rates, latency, errors. Alert on: queue > 10K, workers < 3, failure rate > 10%.',
  },
  {
    id: 'mc5',
    question: 'How do you secure Celery in production?',
    options: [
      'No security needed',
      'Broker auth + SSL/TLS + Flower auth + rate limiting + input validation',
      'Just use HTTPS',
      'Security not important',
    ],
    correctAnswer: 1,
    explanation:
      'Production security: (1) Broker authentication (Redis password, RabbitMQ user/pass), (2) SSL/TLS for connections, (3) Flower authentication (basic auth/OAuth), (4) Rate limiting (prevent abuse), (5) Input validation (prevent injection). Never expose Celery components publicly without security.',
  },
];
