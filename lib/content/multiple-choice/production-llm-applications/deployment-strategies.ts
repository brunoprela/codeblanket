import { MultipleChoiceQuestion } from '../../../types';

export const deploymentStrategiesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pllm-deploy-mc-1',
    question: 'What is the purpose of Docker health checks?',
    options: [
      'Improve performance',
      'Let orchestrators know if container is healthy and ready for traffic',
      'Save memory',
      'Enable scaling',
    ],
    correctAnswer: 1,
    explanation:
      'Health checks (HEALTHCHECK in Dockerfile, liveness/readiness probes in K8s) tell orchestrators if container is healthy, preventing traffic to unhealthy instances.',
  },
  {
    id: 'pllm-deploy-mc-2',
    question: 'What is a rolling deployment?',
    options: [
      'Deploy to all servers at once',
      'Update servers one at a time while maintaining availability',
      'Backup strategy',
      'Testing in production',
    ],
    correctAnswer: 1,
    explanation:
      'Rolling deployment updates pods/servers one at a time (10% → 30% → 100%), maintaining minimum replicas for zero-downtime updates.',
  },
  {
    id: 'pllm-deploy-mc-3',
    question: 'When should you use blue-green deployment?',
    options: [
      'For all deployments',
      'Never',
      'For major releases requiring thorough testing and instant rollback capability',
      'Only for small changes',
    ],
    correctAnswer: 2,
    explanation:
      'Blue-green maintains two environments, deploy to green, test thoroughly, switch traffic 100%. Best for major releases needing confidence and instant rollback.',
  },
  {
    id: 'pllm-deploy-mc-4',
    question: 'What is a multi-stage Docker build?',
    options: [
      'Building multiple images',
      'Using builder stage for dependencies, then copying to smaller runtime stage',
      'Sequential builds',
      'Parallel builds',
    ],
    correctAnswer: 1,
    explanation:
      'Multi-stage builds use builder stage with all build tools, then copy only artifacts to slim runtime stage, dramatically reducing image size.',
  },
  {
    id: 'pllm-deploy-mc-5',
    question: 'How should secrets be managed in containers?',
    options: [
      'Hardcoded in Dockerfile',
      'In environment variables from secrets management (AWS Secrets Manager, K8s Secrets)',
      'In code',
      'Config files',
    ],
    correctAnswer: 1,
    explanation:
      'Never hardcode secrets. Use environment variables populated from secrets management systems, never commit secrets to git or include in images.',
  },
];
