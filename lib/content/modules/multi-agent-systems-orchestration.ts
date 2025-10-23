/**
 * Multi-Agent Systems & Orchestration Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { multiagentarchitecturefundamentalsSection } from '../sections/multi-agent-systems-orchestration/multi-agent-architecture-fundamentals';
import { agentcommunicationprotocolsSection } from '../sections/multi-agent-systems-orchestration/agent-communication-protocols';
import { specializedagentsSection } from '../sections/multi-agent-systems-orchestration/specialized-agents';
import { taskdecompositionplanningSection } from '../sections/multi-agent-systems-orchestration/task-decomposition-planning';
import { agentcoordinationstrategiesSection } from '../sections/multi-agent-systems-orchestration/agent-coordination-strategies';
import { multiagentworkflowsSection } from '../sections/multi-agent-systems-orchestration/multi-agent-workflows';
import { interagentmemorystateSection } from '../sections/multi-agent-systems-orchestration/inter-agent-memory-state';
import { langgraphagentorchestrationSection } from '../sections/multi-agent-systems-orchestration/langgraph-agent-orchestration';
import { crewaiagentframeworksSection } from '../sections/multi-agent-systems-orchestration/crewai-agent-frameworks';
import { multiagentdebuggingSection } from '../sections/multi-agent-systems-orchestration/multi-agent-debugging';
import { humaninloopagentsSection } from '../sections/multi-agent-systems-orchestration/human-in-loop-agents';
import { buildingproductionmultiagentsystemsSection } from '../sections/multi-agent-systems-orchestration/building-production-multi-agent-systems';

// Import quizzes
import { multiagentarchitecturefundamentalsQuiz } from '../quizzes/multi-agent-systems-orchestration/multi-agent-architecture-fundamentals';
import { agentcommunicationprotocolsQuiz } from '../quizzes/multi-agent-systems-orchestration/agent-communication-protocols';
import { specializedagentsQuiz } from '../quizzes/multi-agent-systems-orchestration/specialized-agents';
import { taskdecompositionplanningQuiz } from '../quizzes/multi-agent-systems-orchestration/task-decomposition-planning';
import { agentcoordinationstrategiesQuiz } from '../quizzes/multi-agent-systems-orchestration/agent-coordination-strategies';
import { multiagentworkflowsQuiz } from '../quizzes/multi-agent-systems-orchestration/multi-agent-workflows';
import { interagentmemorystateQuiz } from '../quizzes/multi-agent-systems-orchestration/inter-agent-memory-state';
import { langgraphagentorchestrationQuiz } from '../quizzes/multi-agent-systems-orchestration/langgraph-agent-orchestration';
import { crewaiagentframeworksQuiz } from '../quizzes/multi-agent-systems-orchestration/crewai-agent-frameworks';
import { multiagentdebuggingQuiz } from '../quizzes/multi-agent-systems-orchestration/multi-agent-debugging';
import { humaninloopagentsQuiz } from '../quizzes/multi-agent-systems-orchestration/human-in-loop-agents';
import { buildingproductionmultiagentsystemsQuiz } from '../quizzes/multi-agent-systems-orchestration/building-production-multi-agent-systems';

// Import multiple choice
import { multiagentarchitecturefundamentalsMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/multi-agent-architecture-fundamentals';
import { agentcommunicationprotocolsMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/agent-communication-protocols';
import { specializedagentsMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/specialized-agents';
import { taskdecompositionplanningMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/task-decomposition-planning';
import { agentcoordinationstrategiesMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/agent-coordination-strategies';
import { multiagentworkflowsMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/multi-agent-workflows';
import { interagentmemorystateMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/inter-agent-memory-state';
import { langgraphagentorchestrationMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/langgraph-agent-orchestration';
import { crewaiagentframeworksMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/crewai-agent-frameworks';
import { multiagentdebuggingMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/multi-agent-debugging';
import { humaninloopagentsMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/human-in-loop-agents';
import { buildingproductionmultiagentsystemsMultipleChoice } from '../multiple-choice/multi-agent-systems-orchestration/building-production-multi-agent-systems';

export const multiAgentSystemsOrchestrationModule: Module = {
  id: 'applied-ai-multi-agent',
  title: 'Multi-Agent Systems & Orchestration',
  description:
    'Master building complex AI systems with multiple specialized agents working together: from agent communication and coordination to production multi-agent architectures that solve sophisticated problems collaboratively.',
  category: 'Applied AI',
  difficulty: 'Advanced',
  estimatedTime: '18 hours',
  prerequisites: [
    'LLM Engineering Fundamentals',
    'Tool Use & Function Calling',
    'Python proficiency',
  ],
  icon: '🤝',
  keyTakeaways: [
    'Design multi-agent architectures for complex problem-solving',
    'Implement effective agent communication protocols',
    'Build specialized agents with distinct capabilities',
    'Decompose complex tasks into agent-executable subtasks',
    'Coordinate multiple agents for parallel and sequential execution',
    'Create robust multi-agent workflows with error handling',
    'Manage shared and private memory across agents',
    'Use LangGraph for sophisticated agent orchestration',
    'Leverage CrewAI and other frameworks for rapid development',
    'Debug complex multi-agent interactions systematically',
    'Integrate human oversight and approval workflows',
    'Deploy production multi-agent systems at scale',
  ],
  learningObjectives: [
    'Understand when and why to use multiple agents vs single agent',
    'Design agent communication patterns and message protocols',
    'Create specialized agents for research, coding, planning, and execution',
    'Implement task decomposition and planning algorithms',
    'Build coordination strategies for sequential, parallel, and hierarchical execution',
    'Design state machines and DAG-based workflows for agents',
    'Manage inter-agent memory, state synchronization, and conflict resolution',
    'Master LangGraph for graph-based agent workflows',
    'Compare and use frameworks like CrewAI, AutoGPT, and LangGraph',
    'Trace agent interactions, identify bottlenecks, and debug failures',
    'Implement human-in-the-loop patterns with confidence thresholds',
    'Architect, scale, and monitor production multi-agent systems',
  ],
  sections: [
    {
      ...multiagentarchitecturefundamentalsSection,
      quiz: multiagentarchitecturefundamentalsQuiz,
      multipleChoice: multiagentarchitecturefundamentalsMultipleChoice,
    },
    {
      ...agentcommunicationprotocolsSection,
      quiz: agentcommunicationprotocolsQuiz,
      multipleChoice: agentcommunicationprotocolsMultipleChoice,
    },
    {
      ...specializedagentsSection,
      quiz: specializedagentsQuiz,
      multipleChoice: specializedagentsMultipleChoice,
    },
    {
      ...taskdecompositionplanningSection,
      quiz: taskdecompositionplanningQuiz,
      multipleChoice: taskdecompositionplanningMultipleChoice,
    },
    {
      ...agentcoordinationstrategiesSection,
      quiz: agentcoordinationstrategiesQuiz,
      multipleChoice: agentcoordinationstrategiesMultipleChoice,
    },
    {
      ...multiagentworkflowsSection,
      quiz: multiagentworkflowsQuiz,
      multipleChoice: multiagentworkflowsMultipleChoice,
    },
    {
      ...interagentmemorystateSection,
      quiz: interagentmemorystateQuiz,
      multipleChoice: interagentmemorystateMultipleChoice,
    },
    {
      ...langgraphagentorchestrationSection,
      quiz: langgraphagentorchestrationQuiz,
      multipleChoice: langgraphagentorchestrationMultipleChoice,
    },
    {
      ...crewaiagentframeworksSection,
      quiz: crewaiagentframeworksQuiz,
      multipleChoice: crewaiagentframeworksMultipleChoice,
    },
    {
      ...multiagentdebuggingSection,
      quiz: multiagentdebuggingQuiz,
      multipleChoice: multiagentdebuggingMultipleChoice,
    },
    {
      ...humaninloopagentsSection,
      quiz: humaninloopagentsQuiz,
      multipleChoice: humaninloopagentsMultipleChoice,
    },
    {
      ...buildingproductionmultiagentsystemsSection,
      quiz: buildingproductionmultiagentsystemsQuiz,
      multipleChoice: buildingproductionmultiagentsystemsMultipleChoice,
    },
  ],
};
