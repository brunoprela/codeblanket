/**
 * Multiple choice questions for Alerting Strategies section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const alertingStrategiesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is alert fatigue, and what causes it?',
    options: [
      'Physical tiredness from responding to alerts',
      'When engineers receive so many alerts that they ignore them, missing critical issues',
      'When alerts stop working due to system overload',
      'When alert systems need maintenance',
    ],
    correctAnswer: 1,
    explanation:
      'Alert fatigue occurs when engineers receive so many alerts that they become desensitized and ignore them, potentially missing critical issues. Causes: Too many alerts (50+/day), non-actionable alerts (FYI info), false positives (alerts but no real problem), no aggregation (same issue triggers 100 alerts), wrong severity (everything marked critical). Prevention: Only alert on actionable issues, aggregate related alerts, tune thresholds, classify severity correctly. Target: <5 pages/week per engineer.',
  },
  {
    id: 'mc2',
    question: 'What does "alert on symptoms, not causes" mean?',
    options: [
      'Alert on infrastructure metrics like CPU usage',
      'Alert on user-impacting issues like error rate and latency',
      'Alert on both symptoms and causes equally',
      'Never alert on infrastructure',
    ],
    correctAnswer: 1,
    explanation:
      'Alerting on symptoms means alerting on what users experience (error rate > 1%, latency > 500ms, availability < 99.9%), not infrastructure metrics (CPU > 80%, memory > 70%). Why: High CPU might be fine (batch job), but users care about their experience. Exception: Predictive causes that forecast user impact (disk 95% full will cause failure). Example: Symptom alert: "API error rate 5%" (users affected, action: investigate). Cause alert: "CPU 90%" (unclear if users affected, unclear action).',
  },
  {
    id: 'mc3',
    question: 'What is the purpose of alert duration/threshold configuration?',
    options: [
      'To make alerts slower',
      'To filter transient blips and reduce false positives while maintaining fast detection',
      'To delay all alerts by a fixed amount',
      'To increase alert volume',
    ],
    correctAnswer: 1,
    explanation:
      'Alert duration/threshold filters transient blips while maintaining fast detection. Example: "Alert if error_rate > 5% for 5 minutes with 3 consecutive failures." Duration (5 min) filters brief spikes (deployment blip, single error). Threshold (3 consecutive) requires sustained issue (not one-time anomaly). Trade-off: Too aggressive (30s, 1 failure) = false positives. Too lenient (30 min, 10 failures) = slow detection. Balanced: 3-5 min duration, 2-3 consecutive failures for production APIs.',
  },
  {
    id: 'mc4',
    question: 'What should every alert include to be actionable?',
    options: [
      'Only the metric value',
      "Context (what's wrong, impact, links to dashboards/runbooks)",
      'Just a severity level',
      'Only the alert name',
    ],
    correctAnswer: 1,
    explanation:
      'Actionable alerts include: (1) What\'s wrong ("API error rate > 5%, current: 12%"), (2) Impact ("Affecting 1000 users/min"), (3) When ("Started 5 minutes ago"), (4) Links (Grafana dashboard, Kibana logs, runbook), (5) Possible cause if known ("Recent deployment"), (6) What to do ("Investigate or rollback"). This enables engineers to respond quickly without hunting for information. Bad alert: "API error rate high". Good alert: Full context with links to investigate.',
  },
  {
    id: 'mc5',
    question:
      'How should alerts be grouped/aggregated to prevent alert storms?',
    options: [
      'Send every alert individually',
      'Group related alerts (e.g., "50% of pods unhealthy" instead of 50 separate alerts)',
      'Disable all alerts during incidents',
      'Only send one alert per day',
    ],
    correctAnswer: 1,
    explanation:
      'Alert grouping/aggregation prevents alert storms by combining related alerts. Example: Database down → 50 services alert → Instead of 200 separate alerts, one grouped alert: "Database connectivity issue affecting 50 services." Implementation: Alert dependencies (suppress downstream when upstream fails), deduplication (same issue, same alert once), grouping (combine pod failures into "X% pods unhealthy"). Tools: PagerDuty, Opsgenie support alert grouping. Benefit: Engineers see one alert for root cause instead of 100 symptom alerts.',
  },
];
