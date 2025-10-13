'use client';

import { useStorageInit } from '@/lib/hooks/useStorageInit';

/**
 * Component to initialize storage systems
 * Should be included once in the app layout
 */
export default function StorageInitializer() {
  useStorageInit();
  return null;
}
