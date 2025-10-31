'use client';

/**
 * Data Migration Component
 *
 * Automatically migrates user data from IndexedDB to PostgreSQL when they sign in for the first time.
 * Shows a dialog with migration progress and allows users to skip migration.
 */

import { useEffect, useState } from 'react';
import { useUser } from '@stackframe/stack';
import { migrateToPostgreSQL } from '@/lib/helpers/storage-adapter';
import * as indexedDB from '@/lib/helpers/indexeddb';

export default function DataMigration() {
  const user = useUser();
  const [showMigrationDialog, setShowMigrationDialog] = useState(false);
  const [migrationStatus, setMigrationStatus] = useState<
    'idle' | 'migrating' | 'success' | 'error'
  >('idle');
  const [migrationError, setMigrationError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;

    // Check if migration has already been completed or skipped
    const migrationKey = `migration-completed-${user.id}`;
    const migrationCompleted = localStorage.getItem(migrationKey);

    if (migrationCompleted) {
      return; // Migration already done or skipped
    }

    // Check if there's data to migrate
    checkForDataToMigrate();

    async function checkForDataToMigrate() {
      try {
        const data = await indexedDB.getAllData();
        const dataKeys = Object.keys(data);

        // Only show migration dialog if there's significant data
        // (ignore small amounts like just migration flags)
        if (dataKeys.length > 2) {
          setShowMigrationDialog(true);
        } else {
          // No data to migrate, mark as completed
          if (user) {
            localStorage.setItem(
              `migration-completed-${user.id}`,
              'skipped-no-data',
            );
          }
        }
      } catch (error) {
        console.error('Failed to check for data to migrate:', error);
      }
    }
  }, [user]);

  const handleMigrate = async () => {
    if (!user) return;

    setMigrationStatus('migrating');
    setMigrationError(null);

    try {
      await migrateToPostgreSQL();
      setMigrationStatus('success');
      localStorage.setItem(`migration-completed-${user.id}`, 'completed');

      // Close dialog after 2 seconds
      setTimeout(() => {
        setShowMigrationDialog(false);
      }, 2000);
    } catch (error) {
      console.error('Migration failed:', error);
      setMigrationStatus('error');
      setMigrationError(
        error instanceof Error ? error.message : 'Unknown error',
      );
    }
  };

  const handleSkip = () => {
    if (!user) return;

    localStorage.setItem(`migration-completed-${user.id}`, 'skipped');
    setShowMigrationDialog(false);
  };

  if (!showMigrationDialog) return null;

  return (
    <div className="bg-opacity-50 fixed inset-0 z-50 flex items-center justify-center bg-black p-4">
      <div className="w-full max-w-md rounded-lg bg-[#282a36] p-6 shadow-xl">
        <h2 className="mb-4 text-2xl font-bold text-white">Welcome! 🎉</h2>

        {migrationStatus === 'idle' && (
          <>
            <p className="mb-4 text-gray-300">
              We detected existing progress on this device. Would you like to
              sync it to your account?
            </p>
            <p className="mb-6 text-sm text-gray-400">
              This will upload your completed problems, code solutions, quiz
              progress, and video recordings to the cloud.
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleMigrate}
                className="flex-1 rounded-md bg-[#bd93f9] px-4 py-2 font-medium text-white transition-colors hover:bg-[#a070e0]"
              >
                Sync My Data
              </button>
              <button
                onClick={handleSkip}
                className="flex-1 rounded-md border border-gray-600 px-4 py-2 font-medium text-gray-300 transition-colors hover:bg-[#44475a]"
              >
                Skip
              </button>
            </div>
          </>
        )}

        {migrationStatus === 'migrating' && (
          <div className="text-center">
            <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-t-2 border-b-2 border-[#bd93f9]"></div>
            <p className="text-gray-300">Syncing your data...</p>
            <p className="mt-2 text-sm text-gray-400">This may take a moment</p>
          </div>
        )}

        {migrationStatus === 'success' && (
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500">
              <svg
                className="h-8 w-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
            <p className="text-lg font-medium text-green-400">Sync Complete!</p>
            <p className="mt-2 text-sm text-gray-400">
              Your progress is now saved to your account
            </p>
          </div>
        )}

        {migrationStatus === 'error' && (
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-500">
              <svg
                className="h-8 w-8 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </div>
            <p className="text-lg font-medium text-red-400">Sync Failed</p>
            <p className="mt-2 text-sm text-gray-400">{migrationError}</p>
            <div className="mt-6 flex gap-3">
              <button
                onClick={handleMigrate}
                className="flex-1 rounded-md bg-[#bd93f9] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#a070e0]"
              >
                Try Again
              </button>
              <button
                onClick={handleSkip}
                className="flex-1 rounded-md border border-gray-600 px-4 py-2 text-sm font-medium text-gray-300 transition-colors hover:bg-[#44475a]"
              >
                Skip
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
