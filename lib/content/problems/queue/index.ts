/**
 * queue Problems
 * 8 problems total
 */

import { implement_queueProblem } from './implement-queue';
import { recent_callsProblem } from './recent-calls';
import { circular_queueProblem } from './circular-queue';
import { moving_averageProblem } from './moving-average';
import { perfect_squaresProblem } from './perfect-squares';
import { walls_gatesProblem } from './walls-gates';
import { open_lockProblem } from './open-lock';
import { sliding_window_maxProblem } from './sliding-window-max';

export const queueProblems = [
  implement_queueProblem, // 1. Implement Queue using Stacks
  recent_callsProblem, // 2. Number of Recent Calls
  circular_queueProblem, // 3. Design Circular Queue
  moving_averageProblem, // 4. Moving Average from Data Stream
  perfect_squaresProblem, // 5. Perfect Squares (BFS)
  walls_gatesProblem, // 6. Walls and Gates (Multi-Source BFS)
  open_lockProblem, // 7. Open the Lock (BFS)
  sliding_window_maxProblem, // 8. Sliding Window Maximum
];
