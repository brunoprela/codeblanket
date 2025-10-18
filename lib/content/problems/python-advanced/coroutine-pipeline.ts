/**
 * Data Processing Pipeline with Coroutines
 * Problem ID: coroutine-pipeline
 * Order: 11
 */

import { Problem } from '../../../types';

export const coroutine_pipelineProblem: Problem = {
  id: 'coroutine-pipeline',
  title: 'Data Processing Pipeline with Coroutines',
  difficulty: 'Hard',
  description: `Create a data processing pipeline using coroutines (generators with send()).

Build a pipeline that:
- Accepts data via send()
- Processes data through multiple stages
- Each stage is a coroutine
- Data flows: source -> processor -> sink

**Pattern:**
python
pipeline = source() | process() | sink()
for item in data:
    pipeline.send(item)
`,
  examples: [
    {
      input: 'Numbers 1-5',
      output: 'Processes through pipeline stages',
    },
  ],
  constraints: [
    'Use coroutines (yield with send)',
    'Chain coroutines together',
    'Prime coroutines with next()',
  ],
  hints: [
    'Each coroutine yields then receives',
    'Target coroutine in each stage',
    'Prime with next() before sending',
  ],
  starterCode: `def producer(target):
    """
    Producer coroutine that sends data to target.
    """
    # Your code here
    pass

def processor(target, transform):
    """
    Processor coroutine that transforms and forwards data.
    """
    # Your code here
    pass

def consumer():
    """
    Consumer coroutine that receives and prints data.
    """
    # Your code here
    pass


# Build pipeline: double numbers then print
sink = consumer()
proc = processor(sink, lambda x: x * 2)
source = producer(proc)

for i in range(5):
    source.send(i)
`,
  testCases: [
    {
      input: [[1, 2, 3]],
      expected: [2, 4, 6],
    },
  ],
  solution: `def producer(target):
    while True:
        item = yield
        target.send(item)

def processor(target, transform):
    while True:
        item = yield
        result = transform(item)
        target.send(result)

def consumer():
    while True:
        item = yield
        print(item)

# Prime coroutines
def prime(coro):
    next(coro)
    return coro`,
  timeComplexity: 'O(n) where n is items',
  spaceComplexity: 'O(1)',
  order: 11,
  topic: 'Python Advanced',
};
