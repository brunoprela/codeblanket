export const testCoverageQuiz = [
  {
    id: 'tc-q-1',
    question:
      'Team debate: Should we aim for 100% test coverage? Address: benefits of 100%, diminishing returns, cost-benefit analysis, coverage vs quality, and recommended target with justification.',
    sampleAnswer:
      '100% coverage debate: (1) Benefits: Confidence all code executed, no untested paths, prevents "that will never happen" bugs. (2) Diminishing returns: Last 10% (90%→100%) takes 50% of effort. Trivial code (__repr__, getters) not worth testing. Edge cases rare but time-consuming. (3) Cost: Team spends 2 weeks reaching 100% vs 2 days for 85%. Maintenance: 100% coverage harder to maintain (every trivial change needs test). (4) Coverage ≠ quality: 100% coverage with weak assertions means nothing. Better: 80% with strong assertions than 100% with weak tests. (5) Recommendation: 80-90% target. Critical paths (payment, auth): 95%+. Utilities: 70-80% OK. Trivial code: exclude with pragma: no cover. Focus effort on high-risk areas. 80% scientifically proven to catch most bugs, 100% not worth cost.',
    keyPoints: [
      'Benefits: Confidence, no untested paths, prevents edge case bugs',
      'Diminishing returns: Last 10% takes 50% effort, trivial code not worth testing',
      'Cost: 100% takes weeks vs days for 85%, maintenance burden increases',
      'Coverage ≠ quality: 80% with strong assertions > 100% with weak tests',
      'Recommendation: 80-90% overall, 95%+ critical paths, 70-80% utilities, exclude trivial',
    ],
  },
];
