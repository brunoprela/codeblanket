/**
 * Export/Import functionality for user progress
 */

import { getAllData, importData, migrateFromLocalStorage } from './indexeddb';

export interface ExportData {
    version: string;
    exportDate: string;
    data: Record<string, unknown>;
}

/**
 * Get all data from localStorage that should be backed up
 */
function getLocalStorageData(): Record<string, unknown> {
    const data: Record<string, unknown> = {};
    const prefixes = [
        'codeblanket_completed_problems',
        'codeblanket_code_',
        'module-',
    ];

    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (!key) continue;

        // Skip migration and backup keys
        if (
            key === 'codeblanket_migration_complete' ||
            key === 'codeblanket-auto-backup'
        ) {
            continue;
        }

        // Check if key matches our prefixes
        const shouldInclude = prefixes.some((prefix) => key.startsWith(prefix));

        if (shouldInclude) {
            const value = localStorage.getItem(key);
            if (value) {
                try {
                    data[key] = JSON.parse(value);
                } catch {
                    data[key] = value;
                }
            }
        }
    }

    return data;
}

/**
 * Export all progress data to a JSON file
 */
export async function exportProgress(): Promise<void> {
    try {
        // Get data from both IndexedDB and localStorage
        let data = await getAllData();

        // If IndexedDB is empty or has very little data, use localStorage as fallback
        const indexedDBKeys = Object.keys(data);
        if (indexedDBKeys.length < 2) {
            console.warn(
                'IndexedDB has minimal data, including localStorage data in export',
            );
            const localStorageData = getLocalStorageData();
            data = { ...localStorageData, ...data }; // Merge, preferring IndexedDB
        }

        const exportData: ExportData = {
            version: '1.0',
            exportDate: new Date().toISOString(),
            data,
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json',
        });

        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `codeblanket-progress-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Export error:', error);
        throw error;
    }
}

/**
 * Import progress data from a JSON file
 */
export async function importProgress(file: File): Promise<void> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = async (e) => {
            try {
                const content = e.target?.result as string;
                const importedData: ExportData = JSON.parse(content);

                // Validate the data
                if (!importedData.version || !importedData.data) {
                    throw new Error('Invalid export file format');
                }

                // Import to IndexedDB
                await importData(importedData.data);

                // Also restore to localStorage for immediate access
                Object.entries(importedData.data).forEach(([key, value]) => {
                    try {
                        localStorage.setItem(key, JSON.stringify(value));
                    } catch (error) {
                        console.error(`Failed to restore ${key} to localStorage:`, error);
                    }
                });

                // Trigger storage event to update UI
                window.dispatchEvent(new Event('storage'));

                resolve();
            } catch (error) {
                console.error('Import error:', error);
                reject(error);
            }
        };

        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
}

/**
 * Download a backup of current progress (auto-backup)
 */
export async function createAutoBackup(): Promise<void> {
    try {
        const data = await getAllData();
        const backupKey = 'codeblanket-auto-backup';
        const backup = {
            date: new Date().toISOString(),
            data,
        };

        // Store in localStorage as a last resort backup
        localStorage.setItem(backupKey, JSON.stringify(backup));
    } catch (error) {
        console.error('Auto-backup error:', error);
    }
}

/**
 * Force sync all localStorage data to IndexedDB
 * Useful for debugging or ensuring all data is backed up
 */
export async function forceSyncToIndexedDB(): Promise<void> {
    try {
        // Clear migration flag to force re-migration
        localStorage.removeItem('codeblanket_migration_complete');

        // Run migration
        await migrateFromLocalStorage();

        // Restore migration flag
        localStorage.setItem('codeblanket_migration_complete', 'true');
    } catch (error) {
        console.error('Force sync error:', error);
        throw error;
    }
}
