/**
 * Module system - Educational content for algorithm topics
 */

import { Module, ModuleCategory } from '@/lib/types';
import { advancedGraphsModule } from './advanced-graphs';
import { arraysHashingModule } from './arrays-hashing';
import { backtrackingModule } from './backtracking';
import { binarySearchModule } from './binary-search';
import { bitManipulationModule } from './bit-manipulation';
import { dynamicProgrammingModule } from './dynamic-programming';
import { graphsModule } from './graphs';
import { greedyModule } from './greedy';
import { heapModule } from './heap';
import { intervalsModule } from './intervals';
import { linkedListModule } from './linked-list';
import { mathGeometryModule } from './math-geometry';
import { slidingWindowModule } from './sliding-window';
import { stackModule } from './stack';
import { treesModule } from './trees';
import { triesModule } from './tries';
import { twoPointersModule } from './two-pointers';
import { problemCategories } from '@/lib/problems';

export const allModules: Module[] = [
  advancedGraphsModule,
  arraysHashingModule,
  backtrackingModule,
  binarySearchModule,
  bitManipulationModule,
  dynamicProgrammingModule,
  graphsModule,
  greedyModule,
  heapModule,
  intervalsModule,
  linkedListModule,
  mathGeometryModule,
  slidingWindowModule,
  stackModule,
  treesModule,
  triesModule,
  twoPointersModule,
];

/**
 * Module categories with associated problem counts
 */
export const moduleCategories: ModuleCategory[] = [
  {
    id: advancedGraphsModule.id,
    title: advancedGraphsModule.title,
    description: advancedGraphsModule.description,
    icon: advancedGraphsModule.icon,
    module: advancedGraphsModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'advanced-graphs')?.problems
        .length || 0,
  },
  {
    id: arraysHashingModule.id,
    title: arraysHashingModule.title,
    description: arraysHashingModule.description,
    icon: arraysHashingModule.icon,
    module: arraysHashingModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'arrays-hashing')?.problems
        .length || 0,
  },
  {
    id: backtrackingModule.id,
    title: backtrackingModule.title,
    description: backtrackingModule.description,
    icon: backtrackingModule.icon,
    module: backtrackingModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'backtracking')?.problems
        .length || 0,
  },
  {
    id: binarySearchModule.id,
    title: binarySearchModule.title,
    description: binarySearchModule.description,
    icon: binarySearchModule.icon,
    module: binarySearchModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'binary-search')?.problems
        .length || 0,
  },
  {
    id: bitManipulationModule.id,
    title: bitManipulationModule.title,
    description: bitManipulationModule.description,
    icon: bitManipulationModule.icon,
    module: bitManipulationModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'bit-manipulation')?.problems
        .length || 0,
  },
  {
    id: dynamicProgrammingModule.id,
    title: dynamicProgrammingModule.title,
    description: dynamicProgrammingModule.description,
    icon: dynamicProgrammingModule.icon,
    module: dynamicProgrammingModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'dynamic-programming')
        ?.problems.length || 0,
  },
  {
    id: graphsModule.id,
    title: graphsModule.title,
    description: graphsModule.description,
    icon: graphsModule.icon,
    module: graphsModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'graphs')?.problems.length ||
      0,
  },
  {
    id: greedyModule.id,
    title: greedyModule.title,
    description: greedyModule.description,
    icon: greedyModule.icon,
    module: greedyModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'greedy')?.problems.length ||
      0,
  },
  {
    id: heapModule.id,
    title: heapModule.title,
    description: heapModule.description,
    icon: heapModule.icon,
    module: heapModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'heap')?.problems.length || 0,
  },
  {
    id: intervalsModule.id,
    title: intervalsModule.title,
    description: intervalsModule.description,
    icon: intervalsModule.icon,
    module: intervalsModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'intervals')?.problems
        .length || 0,
  },
  {
    id: linkedListModule.id,
    title: linkedListModule.title,
    description: linkedListModule.description,
    icon: linkedListModule.icon,
    module: linkedListModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'linked-list')?.problems
        .length || 0,
  },
  {
    id: mathGeometryModule.id,
    title: mathGeometryModule.title,
    description: mathGeometryModule.description,
    icon: mathGeometryModule.icon,
    module: mathGeometryModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'math-geometry')?.problems
        .length || 0,
  },
  {
    id: slidingWindowModule.id,
    title: slidingWindowModule.title,
    description: slidingWindowModule.description,
    icon: slidingWindowModule.icon,
    module: slidingWindowModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'sliding-window')?.problems
        .length || 0,
  },
  {
    id: stackModule.id,
    title: stackModule.title,
    description: stackModule.description,
    icon: stackModule.icon,
    module: stackModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'stack')?.problems.length || 0,
  },
  {
    id: treesModule.id,
    title: treesModule.title,
    description: treesModule.description,
    icon: treesModule.icon,
    module: treesModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'trees')?.problems.length || 0,
  },
  {
    id: triesModule.id,
    title: triesModule.title,
    description: triesModule.description,
    icon: triesModule.icon,
    module: triesModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'tries')?.problems.length || 0,
  },
  {
    id: twoPointersModule.id,
    title: twoPointersModule.title,
    description: twoPointersModule.description,
    icon: twoPointersModule.icon,
    module: twoPointersModule,
    problemCount:
      problemCategories.find((cat) => cat.id === 'two-pointers')?.problems
        .length || 0,
  },
];

/**
 * Get module by ID
 */
export function getModuleById(id: string): Module | undefined {
  return allModules.find((module) => module.id === id);
}

/**
 * Get all modules
 */
export function getAllModules(): Module[] {
  return allModules;
}
