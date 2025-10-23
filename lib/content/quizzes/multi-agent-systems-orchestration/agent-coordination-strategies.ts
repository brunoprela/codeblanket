/**
 * Quiz questions for Agent Coordination Strategies section
 */

export const agentcoordinationstrategiesQuiz = [
  {
    id: 'maas-coord-q-1',
    question:
      "You have 5 agents that need to analyze a dataset. Compare the total execution time and resource utilization for sequential vs parallel coordination. If each agent takes 10 minutes sequentially, but agents share a rate-limited API (max 2 concurrent), what's the optimal strategy?",
    hint: 'Consider the constraint of the rate-limited API.',
    sampleAnswer:
      '**Sequential Coordination:** Agents run one after another. Total time = 5 agents × 10 minutes = 50 minutes. Resource utilization: Only 1 agent working at a time = 20% utilization (1/5 agents busy). API usage: Never hits rate limit (only 1 request at a time). **Parallel Coordination (Naive):** All 5 agents start simultaneously. Expected time = 10 minutes (if no constraints). However, API rate limit is 2 concurrent. Result: First 2 agents start immediately. Other 3 wait. After 10 min: 2 complete, 2 more start. After 20 min: 4 complete, 1 starts. After 30 min: All complete. Actual time = 30 minutes (not 10). Resource utilization: First 10 min: 40% (2/5 busy). Next 10 min: 40%. Last 10 min: 20%. Average: 33%. **Optimal Strategy - Batched Parallel:** Deliberately batch into groups respecting rate limit. Batch 1: Agents 1-2 (parallel). Batch 2: Agents 3-4 (parallel). Batch 3: Agent 5. Total time = 30 minutes (same as naive parallel). But controlled, predictable. No API rejections/retries. **Why Not Just Sequential?:** Sequential takes 50 min vs 30 min. Worth the coordination complexity to save 20 minutes. **Improvements:** (1) If tasks differ in size (some take 5 min, some 15 min), schedule smartly: Pair long + short tasks to minimize total time. (2) Dynamic scheduling: As each agent finishes, immediately start next waiting agent. (3) If API supports bursting (allows 5 concurrent for short bursts), could start all 5 and rely on retry logic, but risky. **Answer:** Optimal is batched parallel with batch size = 2, total time = 30 minutes, resource utilization = 33% average. This respects API limits while maximizing parallelism. Sequential is simpler but 66% slower.',
    keyPoints: [
      'Sequential: Simple but slow (50 min), no API issues',
      'Naive parallel: Faster but needs to handle rate limits',
      'Optimal: Batched parallel matching rate limit (30 min)',
      'Consider resource constraints when choosing strategy',
    ],
  },
  {
    id: 'maas-coord-q-2',
    question:
      'Design a consensus coordination strategy for a multi-agent decision-making system where 5 agents vote on whether to approve a proposal. How do you handle: (1) tie votes, (2) one agent timing out, (3) conflicting opinions requiring discussion?',
    hint: "Think about voting mechanisms and what happens when consensus isn't immediate.",
    sampleAnswer:
      '**Basic Consensus Mechanism:** (1) Gather votes from all 5 agents: approve/reject/abstain. (2) Require majority (≥3/5) to approve. (3) If <3 approve, reject. **Handling Tie Votes (2 approve, 2 reject, 1 abstain):** Strategy A: Designated decider. One agent (or human) has tie-breaking authority. Clear but centralizes power. Strategy B: Require supermajority (4/5 to approve). Ties default to reject (safer). Strategy C: Iteration. When tied, agents see each other\'s reasoning, revote. **Handling Agent Timeout:** (1) Set timeout (e.g., 30 seconds per agent). (2) If agent doesn\'t respond: Option A: Count as abstain. Proceed with 4 votes. Need 3/4 for approval. Option B: Retry once after 10 seconds. If still no response, exclude from vote. Option C: Use agent\'s historical voting pattern (if usually approves 80% of time, count as 0.8 approval). (3) If multiple agents timeout: If ≥3 timeout: Abort, reschedule vote. If 1-2 timeout: Proceed with remaining votes. **Handling Conflicting Opinions (Requires Discussion):** Phase 1 - Initial Vote: All agents vote with brief reasoning. Phase 2 - Discussion (if no consensus): Agents see all votes and reasoning. Agents identify disagreements: "I voted reject because of security concern X, but Agent B voted approve. Agent B, how do you address X?" Agents respond to each other\'s concerns. Set max 2 rounds of discussion (prevent infinite debate). Phase 3 - Final Vote: After discussion, agents revote. If still no consensus → escalate to human. **Implementation:** async function consensus_vote(agents, proposal): votes = gather_votes_with_timeout(agents, timeout=30). if not enough votes (due to timeouts): return "INSUFFICIENT_VOTES". initial_result = tally_votes(votes). if has_supermajority(initial_result): return initial_result. else: # Need discussion, discussion = await discussion_round(agents, votes). final_votes = gather_votes_with_timeout(agents, timeout=30). final_result = tally_votes(final_votes). if has_supermajority(final_result): return final_result. return "ESCALATE_TO_HUMAN". **Best Practices:** Clear voting rules communicated upfront. Timeouts with reasonable retry. Discussion limited to 1-2 rounds. Final decision by supermajority or escalation. Audit trail of all votes and reasoning for later review.',
    keyPoints: [
      'Tie votes: use designated decider, supermajority, or iteration',
      'Timeouts: retry once, then proceed without or count as abstain',
      'Conflicting opinions: enable structured discussion rounds',
      'Always have escalation path to human for unresolved conflicts',
    ],
  },
  {
    id: 'maas-coord-q-3',
    question:
      'Your hierarchical coordination system has a manager agent that becomes a bottleneck as the number of worker agents scales. Redesign the system to eliminate this bottleneck while maintaining coordination.',
    hint: 'Consider multiple managers or decentralized coordination.',
    sampleAnswer:
      '**Problem Analysis:** Single manager coordinates all N workers. Manager must: (1) Receive requests, (2) Decompose into tasks, (3) Assign to workers, (4) Track progress, (5) Aggregate results. As N grows, manager spends more time on 3, 4, 5. Manager becomes bottleneck when its work > worker\'s work. **Solution 1 - Multiple Managers (Hierarchical Layers):** Create middle managers. Top manager coordinates 3-5 middle managers. Each middle manager coordinates 5-10 workers. Example: 30 workers → 1 top + 5 middle (each managing 6 workers). Top manager only talks to 5 agents, not 30. Scales to ~100 workers before adding another layer. Downside: Added latency (3 hops instead of 2). **Solution 2 - Domain Managers:** Divide by domain rather than arbitrary groups. Backend manager, Frontend manager, Database manager. Each manages workers in their domain. Top manager routes tasks to appropriate domain manager. Example: "API task" → Backend manager. "UI task" → Frontend manager. Advantage: Domain expertise at middle layer. **Solution 3 - Worker Self-Coordination (Peer-to-Peer):** Workers pull tasks from shared queue (manager populates queue). Workers report completion to shared state, not manager. Workers coordinate directly with dependencies. Manager only: Creates initial task queue, monitors overall progress, intervenes if stuck. Workers autonomously: Pull next task when free, execute, mark complete, start next. Advantage: No manager bottleneck. Workers scale infinitely. Disadvantage: Requires robust shared state and coordination protocol. **Solution 4 - Leader Election:** Workers elect a temporary leader for each project. Leader coordinates that project. Next project, different leader. Distributes management work across all agents. **Best Solution - Hybrid (Solution 3 with Solution 1 backup):** Primary: Worker self-coordination with task queue. If system grows very large (>50 workers): Add middle managers to manage queue, but workers still self-serve. Implementation: class WorkerSelfCoord: shared_task_queue. shared_state (completed tasks). async worker_loop(worker): while True: task = pull_task(queue). if no task: break. result = await worker.execute(task). shared_state.mark_complete(task, result). class Manager: populate_queue(tasks). monitor_progress(). intervene_if_stuck(). **Scaling:** With self-coordination, system scales to 100s of workers. Manager only creates initial plan and monitors. No per-worker communication needed.',
    keyPoints: [
      'Single manager becomes bottleneck as workers scale',
      'Solutions: multiple managers, domain-based managers, or self-coordination',
      'Worker self-coordination with shared queue scales best',
      'Manager role shifts from coordination to planning and monitoring',
    ],
  },
];
