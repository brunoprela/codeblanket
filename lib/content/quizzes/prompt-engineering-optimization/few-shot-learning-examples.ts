/**
 * Quiz questions for Few-Shot Learning & Examples section
 */

export const fewshotlearningexamplesQuiz = [
  {
    id: 'peo-fewshot-q-1',
    question:
      'What is the optimal number of examples for few-shot prompting? How does this vary by task complexity and model capability? Explain the trade-offs.',
    hint: 'Consider token costs, diminishing returns, task difficulty, and model intelligence.',
    sampleAnswer:
      'Optimal example count depends on three factors: 1) TASK COMPLEXITY: Simple tasks (sentiment analysis) need 1-3 examples; medium tasks (entity extraction) need 3-5; complex tasks (code generation) need 5-10; rarely need more than 10-15 examples; 2) MODEL CAPABILITY: GPT-4 learns faster from fewer examples; weaker models need more demonstrations; 3) CONSISTENCY NEEDS: Production systems requiring high reliability benefit from more examples. TRADE-OFFS: More examples = higher token costs, slower responses, but better consistency and edge case handling. Fewer examples = cheaper, faster, but potentially more variance. In practice: Start with 3-5 examples, A/B test with different counts, monitor failure modes to decide if more needed. For production, consistency often outweighs cost, so err toward more examples. Rule of thumb: use minimum number that achieves 95%+ reliability on test cases.',
    keyPoints: [
      'Simple tasks: 1-3, complex tasks: 5-10 examples',
      'More examples = higher cost but better consistency',
      'Powerful models need fewer examples',
      'Production favors consistency over cost savings',
      'Test to find optimal count for your task',
      'Diminishing returns after 10-15 examples',
    ],
  },
  {
    id: 'peo-fewshot-q-2',
    question:
      'How do you select diverse, representative examples that cover edge cases? What strategies ensure good example coverage?',
    hint: 'Think about input variation, output types, edge cases, and systematic selection methods.',
    sampleAnswer:
      "Selecting diverse examples requires systematic approach: 1) INPUT VARIATION: Include shortest, longest, and medium-length inputs; varied formats and styles; edge cases (empty, null, special characters); 2) OUTPUT DIVERSITY: Cover all possible output categories; include rare but important cases; show how to handle ambiguous situations; 3) DIFFICULTY PROGRESSION: Order simple to complex helps model learn pattern; 4) EDGE CASE COVERAGE: Explicitly include common failure modes; boundary conditions; error handling examples; 5) SYSTEMATIC SELECTION: Use clustering to find representative samples; analyze failures to add missing patterns; ensure balanced distribution across categories. STRATEGIES: Start with manual curation of known cases, augment with failure analysis from production, use embedding similarity to ensure coverage of input space, track which examples improve performance most, remove redundant examples that don't add value. Test coverage by measuring performance on held-out diverse test set.",
    keyPoints: [
      'Vary input length, format, complexity',
      'Include edge cases and boundary conditions',
      'Cover all output categories proportionally',
      'Use failure analysis to find gaps',
      'Order examples simple to complex',
      'Test coverage with diverse test set',
    ],
  },
  {
    id: 'peo-fewshot-q-3',
    question:
      'Explain dynamic example selection (RAG for examples). When is this approach better than static examples?',
    hint: 'Consider relevance, memory efficiency, task variety, and implementation complexity.',
    sampleAnswer:
      'Dynamic example selection retrieves most relevant examples for each query using RAG principles: 1) APPROACH: Embed all examples in vector database; embed incoming query; retrieve top-k most similar examples; include in prompt; 2) BENEFITS: Always shows most relevant examples for current input; memory-efficient for large example pools (thousands); adapts to diverse query types; personalizable per user/domain; 3) WHEN BETTER THAN STATIC: Large example database (>20 examples); highly variable input types requiring different demonstrations; personalized applications where examples vary by user; continuous learning scenarios where examples grow over time. IMPLEMENTATION: Use sentence-transformers or OpenAI embeddings for example/query encoding; vector DB like Pinecone/Weaviate for storage; similarity search at query time; cache common patterns for speed. TRADE-OFFS: Adds latency (embedding + search), requires infrastructure, but dramatically improves relevance and scales better. Used in production by Cursor for code examples and by advanced RAG systems.',
    keyPoints: [
      'Retrieves most relevant examples per query',
      'Scales to large example databases',
      'Better for diverse input types',
      'Requires embeddings and vector search',
      'Adds latency but improves relevance',
      'Essential for personalized applications',
    ],
  },
];
