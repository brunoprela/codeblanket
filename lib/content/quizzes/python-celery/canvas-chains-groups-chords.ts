/**
 * Quiz questions for Canvas: Chains, Groups, Chords section
 */

export const canvasChainsGroupsChordsQuiz = [
  {
    id: 'q1',
    question:
      'Design a workflow using canvas that: downloads 100 files in parallel, processes each, then aggregates results and sends an email.',
    sampleAnswer:
      'CANVAS WORKFLOW: ```python from celery import chord, chain @app.task def download_file(url): return download(url) @app.task def process_file(data): return process(data) @app.task def aggregate(results): return sum(results) @app.task def send_email(total): send_mail(f"Total: {total}") # Workflow workflow = chord( [chain(download_file.s(url), process_file.s()) for url in urls], chain(aggregate.s(), send_email.s()) ) ``` FLOW: 100 parallel chains (download→process), Aggregate results, Send email with total.',
    keyPoints: [
      'Chord for parallel + callback',
      'Chain for sequential steps',
      'Aggregate results',
      'Final email notification',
      'Scalable pattern',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between chain, group, and chord with examples.',
    sampleAnswer:
      'CANVAS PRIMITIVES: **Chain (Sequential)**: ```python chain(A.s(), B.s(), C.s())  # A → B → C ``` **Group (Parallel)**: ```python group([A.s(), B.s(), C.s()])  # [A, B, C] ``` **Chord (Parallel + Callback)**: ```python chord([A.s(), B.s(), C.s()], D.s())  # [A,B,C] → D ``` DIFFERENCE: Chain = sequential, Group = parallel (no callback), Chord = parallel with callback.',
    keyPoints: [
      'Chain: Sequential (A→B→C)',
      'Group: Parallel ([A,B,C])',
      'Chord: Parallel + callback',
      'Use case determines choice',
    ],
  },
  {
    id: 'q3',
    question:
      'Implement error handling in a chord where some parallel tasks might fail.',
    sampleAnswer:
      'CHORD ERROR HANDLING: ```python @app.task(bind=True, max_retries=3) def process_item(self, item): try: return process(item) except Exception as exc: raise self.retry(exc=exc) @app.task def combine_results(results): # Filter out failures valid = [r for r in results if r is not None] return sum(valid) # Chord with error handling callback = combine_results.s() header = [process_item.s(i) for i in items] result = chord(header)(callback) ``` HANDLING: Retry failed tasks, Filter None results, Aggregate valid results only.',
    keyPoints: [
      'Retry failed tasks',
      'Filter None results',
      'Aggregate valid only',
      'Graceful degradation',
      'Production-ready error handling',
    ],
  },
];
