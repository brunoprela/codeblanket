/**
 * Quiz questions for Multi-Agent Workflows section
 */

export const multiagentworkflowsQuiz = [
  {
    id: 'maas-workflow-q-1',
    question:
      'Design a state machine workflow for a document review process with states: Draft → Review → Revise → Approved/Rejected. Define the transitions, conditions for each transition, and how to handle infinite revision loops.',
    hint: 'Think about what triggers each transition and how to prevent getting stuck.',
    sampleAnswer:
      '**States:** DRAFT, REVIEW, REVISE, APPROVED, REJECTED. **Transitions:** (1) DRAFT → REVIEW: Triggered when writer completes initial draft. Condition: draft_complete = true. (2) REVIEW → APPROVED: Triggered when reviewer approves. Condition: review_score >= 8.0 AND no critical issues. (3) REVIEW → REJECTED: Triggered when quality too low. Condition: review_score < 5.0 OR critical issues found. (4) REVIEW → REVISE: Triggered when needs improvement but not rejected. Condition: 5.0 <= review_score < 8.0. (5) REVISE → REVIEW: Triggered when revision complete. Condition: revision_complete = true. **Preventing Infinite Loops:** Problem: REVIEW → REVISE → REVIEW → REVISE (infinite). Solutions: (A) **Max Iterations:** Track iteration_count. If iteration_count >= 3, transition to REJECTED or ESCALATE_TO_HUMAN instead of REVISE. (B) **Progress Requirement:** Each revision must improve score. If new_score <= previous_score, reject (not making progress). (C) **Diminishing Returns:** First revision: full review. Second revision: only check previous issues. Third revision: auto-reject if still not passing. (D) **Time Limit:** If in REVIEW→REVISE loop for >30 minutes, escalate to human. **State Machine Definition:** class DocReviewWorkflow: state = DRAFT, iteration_count = 0, review_scores = []. def transition(self, event): if self.state == DRAFT and event == "draft_complete": self.state = REVIEW. elif self.state == REVIEW: if event == "approve": self.state = APPROVED. elif event == "reject": self.state = REJECTED. elif event == "revise": self.iteration_count += 1. if self.iteration_count >= 3: self.state = REJECTED, reason = "Max iterations". else: self.state = REVISE. elif self.state == REVISE and event == "revision_complete": self.state = REVIEW. **Implementation:** async def execute(): state = DRAFT. doc = await writer.create_draft(). state = REVIEW. while state == REVIEW: score, issues = await reviewer.review(doc). if score >= 8: state = APPROVED. break. elif score < 5: state = REJECTED. break. else: iteration_count += 1. if iteration_count >= 3: state = REJECTED, reason = "Max iterations exceeded". break. state = REVISE. feedback = issues. doc = await writer.revise(doc, feedback). state = REVIEW. **Benefits:** Clear state transitions. No ambiguity on next state. Guaranteed termination (max iterations). Can visualize as diagram.',
    keyPoints: [
      'Define explicit states and transition conditions',
      'Add max iteration count to prevent infinite loops',
      'Require progress (improving score) on each iteration',
      'Have escape hatches (escalation, rejection after N tries)',
    ],
  },
  {
    id: 'maas-workflow-q-2',
    question:
      'You have a DAG workflow where Task E depends on both Task B and Task D, but Task D takes 3x longer than Task B. How would you optimize the workflow execution schedule? Show the execution timeline.',
    hint: 'Consider starting Task D as early as possible.',
    sampleAnswer:
      "**Original DAG:** A → B → E, A → C → D → E. Assume: A=5min, B=10min, C=5min, D=30min, E=10min. **Naive Execution:** Sequential: A(5) → B(10) → C(5) → D(30) → E(10) = 60min. **Parallelizing:** A → [B, C in parallel] → D → E. Timeline: t=0-5: A runs. t=5-15: B runs (10min), C also runs (5min, finishes at t=10). t=15-45: D runs (30min). Must wait for B to finish even though C finished earlier. t=45-55: E runs (10min). Total: 55min. **Optimization Insight:** D takes 3x longer than B. D depends on C (5min). C depends on A (5min). D can start at t=10 (after A→C completes). B can start at t=5 (after A completes). If we start D as early as possible, it finishes at t=10+30=40. B finishes at t=5+10=15. E can start at t=40 (when both B and D complete). **Optimized Schedule:** t=0-5: A runs. t=5-10: C runs (waiting for A). t=5-15: B runs (starts with C, finishes after). t=10-40: D runs (starts as soon as C finishes). t=40-50: E runs (waits for both B@15 and D@40, so starts at t=40). Total: 50min (5min saved). **Key Insight:** Start longest-path tasks first. Critical path is A→C→D→E (50min). Secondary path is A→B (15min). Since D is the bottleneck, start it ASAP. B can wait. **Implementation:** def schedule_dag(tasks): # Find critical path (longest), longest = find_critical_path(tasks). # Sort tasks by: (1) critical path first, (2) duration desc. sorted_tasks = sort_by_priority(tasks). # Execute ready tasks (dependencies met), ready = get_ready(sorted_tasks). **Algorithm - Earliest Start Time:** For each task: earliest_start = max(finish_time of all dependencies). Start each task at its earliest_start. If multiple tasks ready at same time, prioritize by: (1) Longest duration first, (2) On critical path. **Result:** By starting D early (t=10 instead of t=15), we save 5 minutes. E must wait for slowest dependency (D), so minimizing D's finish time is key.",
    keyPoints: [
      'Identify critical path (longest dependency chain)',
      'Start tasks on critical path as early as possible',
      'Prioritize long-duration tasks over short ones',
      'Multiple ready tasks: start longest-running first',
    ],
  },
  {
    id: 'maas-workflow-q-3',
    question:
      'Design a conditional workflow that routes content creation to different agent pipelines based on content type: (code → review → test), (article → edit → fact-check), (image → upscale → tag). How do you handle hybrid content (article with code snippets)?',
    hint: 'Think about detecting content type and handling mixed types.',
    sampleAnswer:
      '**Type Detection:** First, determine content type. async def detect_type(content): if has_code_blocks(content): return "code". elif is_text_document(content): return "article". elif is_image(content): return "image". elif is_mixed(content): return "mixed". **Workflow Routing:** Based on type, route to pipeline. async def route_workflow(content): content_type = detect_type(content). if content_type == "code": return code_pipeline(content). elif content_type == "article": return article_pipeline(content). elif content_type == "image": return image_pipeline(content). elif content_type == "mixed": return mixed_pipeline(content). **Pipeline Definitions:** CODE: code → code_reviewer → tester → approve. ARTICLE: article → editor → fact_checker → approve. IMAGE: image → upscaler → tagger → approve. **Handling Mixed Content (Article + Code):** Strategy 1 - Sequential: Run article pipeline, then code pipeline. Result: Edited article + reviewed code. Combine results. Strategy 2 - Parallel: Split content into article parts and code parts. Run both pipelines in parallel. Merge results. Strategy 3 - Specialized Mixed Pipeline: mixed_content → separator (splits into article + code). article_parts → article_pipeline. code_parts → code_pipeline. merger (combines results). mixed_content → final_reviewer (checks coherence). **Implementation:** async def mixed_pipeline(content): # Split, article_parts, code_parts = split_content(content). # Parallel processing, results = await asyncio.gather(article_pipeline(article_parts), code_pipeline(code_parts)). # Merge, merged = merge_results(results). # Final review for coherence, final = await coherence_reviewer(merged). return final. **Split Logic:** def split_content(content): article_parts = extract_text(content). code_parts = extract_code_blocks(content). return article_parts, code_parts. **Merge Logic:** def merge_results(results): article_result, code_result = results. # Reinsert code blocks into article, merged = article_result. for i, code in enumerate(code_result.reviewed_blocks): merged = insert_code_at_position(merged, code, position=i). return merged. **Benefits:** Pure types: Simple routing. Mixed types: Handled by specialized pipeline. Each part gets appropriate review. Final coherence check ensures parts work together. **Edge Cases:** What if code is 90% of content? Route to code pipeline. What if just one code snippet in article? Use article pipeline but add code review step. Use content_ratio to decide: if code_ratio > 0.5: primary=code, secondary=article. else: primary=article, secondary=code.',
    keyPoints: [
      'Detect content type before routing to pipeline',
      'Define separate pipelines for each content type',
      'Mixed content: split, process in parallel, merge results',
      'Add final coherence check for mixed content',
    ],
  },
];
