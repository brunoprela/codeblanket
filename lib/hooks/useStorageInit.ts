'use client';

import { useEffect } from 'react';
import { migrateFromLocalStorage } from '@/lib/helpers/indexeddb';
import { createAutoBackup } from '@/lib/helpers/export-import';

/**
 * Hook to initialize storage systems
 * - Migrates data from localStorage to IndexedDB on first load
 * - Creates periodic auto-backups
 */
export function useStorageInit() {
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        // Check if migration has already been done
        const migrationKey = 'codeblanket_migration_complete';
        const migrationComplete = localStorage.getItem(migrationKey);

        if (!migrationComplete && mounted) {
          console.debug('Migrating data from localStorage to IndexedDB...');
          await migrateFromLocalStorage();
          localStorage.setItem(migrationKey, 'true');
          console.debug('Migration complete!');
        }

        // Create an auto-backup every time the app loads
        if (mounted) {
          await createAutoBackup();
        }
      } catch (error) {
        console.error('Storage initialization error:', error);
      }
    }

    init();

    // Create auto-backup every 5 minutes
    const backupInterval = setInterval(
      () => {
        if (mounted) {
          createAutoBackup().catch((err) =>
            console.error('Auto-backup failed:', err),
          );
        }
      },
      5 * 60 * 1000,
    ); // 5 minutes

    return () => {
      mounted = false;
      clearInterval(backupInterval);
    };
  }, []);
}
