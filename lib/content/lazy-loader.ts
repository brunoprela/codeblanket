/**
 * Lazy content loader - loads modules on demand instead of all at once
 * This significantly reduces initial bundle size
 */

import { Module } from '@/lib/types';

// Cache for loaded modules
const moduleCache: Map<string, Module> = new Map();

/**
 * Dynamically import and load a module by ID
 * @param moduleId - The module identifier
 * @returns The loaded module or null if not found
 */
export async function loadModuleById(moduleId: string): Promise<Module | null> {
  // Check cache first
  if (moduleCache.has(moduleId)) {
    console.log(`[LazyLoader] Returning cached module: ${moduleId}`);
    return moduleCache.get(moduleId)!;
  }

  console.log(`[LazyLoader] Loading module dynamically: ${moduleId}`);

  try {
    // Dynamically import the module based on ID
    const module = await import(`@/lib/content/modules/${moduleId}`);
    const moduleData = module.default || module[moduleId];

    if (moduleData) {
      // Cache the result
      moduleCache.set(moduleId, moduleData);
      return moduleData;
    }

    console.error(`[LazyLoader] Module ${moduleId} not found in import`);
    return null;
  } catch (error) {
    console.error(`[LazyLoader] Failed to load module ${moduleId}:`, error);
    return null;
  }
}

/**
 * Get module metadata without loading full content
 * @param moduleId - The module identifier
 * @returns Basic module info (id, title, description) without sections
 */
export function getModuleMetadata(moduleId: string): {
  id: string;
  title: string;
  description: string;
} | null {
  // This would need to be generated from your modules
  // For now, return null and we'll load full module
  return null;
}

/**
 * Clear the module cache (useful for testing/development)
 */
export function clearModuleCache(): void {
  moduleCache.clear();
  console.log('[LazyLoader] Module cache cleared');
}
