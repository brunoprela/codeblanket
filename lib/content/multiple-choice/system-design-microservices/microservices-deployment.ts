/**
 * Multiple choice questions for Microservices Deployment section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const microservicesdeploymentMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc-deploy-1',
    question:
      'What is the main advantage of containerization for microservices?',
    options: [
      'Containers are faster than virtual machines',
      'Containers provide consistency across environments and isolation between services',
      'Containers eliminate the need for testing',
      'Containers are only useful in production',
    ],
    correctAnswer: 1,
    explanation:
      'Containers provide consistency ("works on my machine" → works everywhere) and isolation (dependencies don\'t conflict). Package service with all dependencies, run same image in dev/staging/prod. Docker image includes OS, runtime, libraries, code. Option 1 is partially true but not the main advantage. Option 3 is wrong (testing still needed). Option 4 is wrong (containers used in all environments).',
  },
  {
    id: 'mc-deploy-2',
    question: 'What is a canary deployment?',
    options: [
      'Deploying to a bird-themed environment',
      'Gradually shifting traffic to new version (10% → 25% → 50% → 100%)',
      'Deploying all at once',
      'Only deploying to production',
    ],
    correctAnswer: 1,
    explanation:
      'Canary deployment gradually shifts traffic to new version while monitoring metrics. Start with 10% of traffic to new version, monitor for errors/performance, gradually increase if all is well. Named after "canary in coal mine" (early warning). This minimizes risk by testing with real users before full rollout. Requires service mesh (Istio) or load balancer that supports weighted routing. Option 1 is a joke. Option 3 is big bang. Option 4 doesn\'t describe a strategy.',
  },
  {
    id: 'mc-deploy-3',
    question:
      'What is the difference between a liveness probe and a readiness probe?',
    options: [
      'They are the same thing',
      'Liveness restarts container if fails; Readiness removes from load balancer if fails',
      'Liveness is for production; Readiness is for development',
      'Readiness restarts container; Liveness removes from load balancer',
    ],
    correctAnswer: 1,
    explanation:
      "Liveness probe checks if container is alive (e.g., /health endpoint). If fails, Kubernetes RESTARTS the container (fixes deadlocks, crashes). Readiness probe checks if container is ready for traffic (e.g., /ready endpoint checking database connection). If fails, Kubernetes REMOVES pod from service load balancer but doesn't restart (temporary issue, will recover). Need both for proper health management. Option 1 is wrong (different purposes). Option 3 is wrong (both used in all environments). Option 4 is backwards.",
  },
  {
    id: 'mc-deploy-4',
    question:
      'Why must database migrations be backward-compatible in microservices?',
    options: [
      'For better performance',
      'Because old and new service versions coexist during rolling deployment',
      'To reduce database size',
      'For security reasons',
    ],
    correctAnswer: 1,
    explanation:
      "During rolling deployment, old and new service versions run simultaneously. If you add a NOT NULL column, old service version (doesn't know about it) will fail to write to database. Solution: backward-compatible migrations in steps - add as nullable, deploy service, backfill, then enforce NOT NULL. Each step works with both old and new service versions. Option 1 is unrelated. Option 3 is unrelated. Option 4 is not the main reason.",
  },
  {
    id: 'mc-deploy-5',
    question:
      'What is the purpose of Horizontal Pod Autoscaler (HPA) in Kubernetes?',
    options: [
      'To manually scale pods',
      'To automatically add/remove pods based on CPU/memory metrics',
      'To rotate pods regularly',
      'To distribute pods across nodes',
    ],
    correctAnswer: 1,
    explanation:
      "HPA automatically scales the number of pods based on metrics (CPU, memory, custom metrics). Define min/max replicas and target utilization (e.g., 70% CPU). Kubernetes adds pods when load increases, removes when load decreases. This handles variable load without manual intervention. Example: Black Friday traffic spike → HPA scales from 3 to 20 pods → traffic drops → scales back to 3. Option 1 is wrong (automatic, not manual). Option 3 is wrong (that's pod disruption budget). Option 4 is wrong (that's node affinity).",
  },
];
