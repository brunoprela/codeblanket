import { Module, ModuleCategory } from '../types';
import { problemCategories } from '../problems';

import { advancedGraphsModule } from './advanced-graphs';
import { arraysHashingModule } from './arrays-hashing';
import { backtrackingModule } from './backtracking';
import { bfsModule } from './bfs';
import { binarySearchModule } from './binary-search';
import { bitManipulationModule } from './bit-manipulation';
import { dfsModule } from './dfs';
import { dynamicProgrammingModule } from './dynamic-programming';
import { fenwickTreeModule } from './fenwick-tree';
import { graphsModule } from './graphs';
import { greedyModule } from './greedy';
import { heapModule } from './heap';
import { intervalsModule } from './intervals';
import { linkedListModule } from './linked-list';
import { mathGeometryModule } from './math-geometry';
import { segmentTreeModule } from './segment-tree';
import { slidingWindowModule } from './sliding-window';
import { stackModule } from './stack';
import { treesModule } from './trees';
import { triesModule } from './tries';
import { twoPointersModule } from './two-pointers';

/**
 * All available modules
 */
export const allModules: Module[] = [
    advancedGraphsModule,
    arraysHashingModule,
    backtrackingModule,
    bfsModule,
    binarySearchModule,
    bitManipulationModule,
    dfsModule,
    dynamicProgrammingModule,
    fenwickTreeModule,
    graphsModule,
    greedyModule,
    heapModule,
    intervalsModule,
    linkedListModule,
    mathGeometryModule,
    segmentTreeModule,
    slidingWindowModule,
    stackModule,
    treesModule,
    triesModule,
    twoPointersModule,
];

/**
 * Module categories with associated problem counts
 * Ordered in logical learning progression from fundamentals to advanced topics
 */
export const moduleCategories: ModuleCategory[] = [
    // 1. Fundamentals - Start with basic data structures
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
    // 2. Simple techniques on arrays
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
    // 3. Core search/traversal algorithms
    {
        id: dfsModule.id,
        title: dfsModule.title,
        description: dfsModule.description,
        icon: dfsModule.icon,
        module: dfsModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'dfs')?.problems.length || 0,
    },
    {
        id: bfsModule.id,
        title: bfsModule.title,
        description: bfsModule.description,
        icon: bfsModule.icon,
        module: bfsModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'bfs')?.problems.length || 0,
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
    // 4. Basic data structures
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
        id: linkedListModule.id,
        title: linkedListModule.title,
        description: linkedListModule.description,
        icon: linkedListModule.icon,
        module: linkedListModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'linked-list')?.problems
                .length || 0,
    },
    // 5. Hierarchical structures
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
        id: heapModule.id,
        title: heapModule.title,
        description: heapModule.description,
        icon: heapModule.icon,
        module: heapModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'heap')?.problems.length || 0,
    },
    // 6. Graph structures and algorithms
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
    // 7. Advanced problem-solving techniques
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
        id: dynamicProgrammingModule.id,
        title: dynamicProgrammingModule.title,
        description: dynamicProgrammingModule.description,
        icon: dynamicProgrammingModule.icon,
        module: dynamicProgrammingModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'dynamic-programming')
                ?.problems.length || 0,
    },
    // 8. Specialized structures and techniques
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
        id: greedyModule.id,
        title: greedyModule.title,
        description: greedyModule.description,
        icon: greedyModule.icon,
        module: greedyModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'greedy')?.problems.length ||
            0,
    },
    // 9. Advanced topics
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
    // 10. Advanced data structures
    {
        id: segmentTreeModule.id,
        title: segmentTreeModule.title,
        description: segmentTreeModule.description,
        icon: segmentTreeModule.icon,
        module: segmentTreeModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'segment-tree')?.problems
                .length || 0,
    },
    {
        id: fenwickTreeModule.id,
        title: fenwickTreeModule.title,
        description: fenwickTreeModule.description,
        icon: fenwickTreeModule.icon,
        module: fenwickTreeModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'fenwick-tree')?.problems
                .length || 0,
    },
    // 11. Bit manipulation and math
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
        id: mathGeometryModule.id,
        title: mathGeometryModule.title,
        description: mathGeometryModule.description,
        icon: mathGeometryModule.icon,
        module: mathGeometryModule,
        problemCount:
            problemCategories.find((cat) => cat.id === 'math-geometry')?.problems
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
