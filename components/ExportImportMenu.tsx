'use client';

import { useState, useRef } from 'react';
import {
  exportProgress,
  importProgress,
  forceSyncToIndexedDB,
} from '@/lib/helpers/export-import';

export default function ExportImportMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [message, setMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = async () => {
    try {
      await exportProgress();
      setMessage({ type: 'success', text: 'Progress exported successfully!' });
      setTimeout(() => setMessage(null), 3000);
      setIsOpen(false);
    } catch (error) {
      console.error('Export failed:', error);
      setMessage({ type: 'error', text: 'Export failed. Please try again.' });
      setTimeout(() => setMessage(null), 3000);
    }
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsImporting(true);
    try {
      await importProgress(file);
      setMessage({
        type: 'success',
        text: 'Import complete! Please refresh the page (Cmd/Ctrl+R) to see your data.',
      });
      setIsImporting(false);
      // Don't auto-reload - let user manually refresh after videos finish importing
      // setTimeout(() => {
      //   window.location.reload();
      // }, 1500);
    } catch (error) {
      console.error('Import failed:', error);
      setMessage({
        type: 'error',
        text: 'Import failed. Please check the file format.',
      });
      setIsImporting(false);
      setTimeout(() => setMessage(null), 3000);
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleForceSync = async () => {
    setIsSyncing(true);
    try {
      await forceSyncToIndexedDB();
      setMessage({
        type: 'success',
        text: 'All data synced to IndexedDB successfully!',
      });
      setTimeout(() => setMessage(null), 3000);
      setIsOpen(false);
    } catch (error) {
      console.error('Sync failed:', error);
      setMessage({
        type: 'error',
        text: 'Sync failed. Please try again.',
      });
      setTimeout(() => setMessage(null), 3000);
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="relative">
      {/* Dropdown Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-lg border-2 border-[#bd93f9] bg-transparent px-4 py-2 text-sm font-semibold text-[#bd93f9] transition-colors hover:bg-[#bd93f9] hover:text-[#282a36]"
        title="Backup & Restore"
      >
        <svg
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"
          />
        </svg>
        <span className="hidden sm:inline">Backup</span>
        <svg
          className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute right-0 z-20 mt-2 w-64 rounded-lg border-2 border-[#44475a] bg-[#282a36] shadow-xl">
            <div className="p-2">
              {/* Export Button */}
              <button
                onClick={handleExport}
                className="flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm font-medium text-[#f8f8f2] transition-colors hover:bg-[#44475a]"
              >
                <svg
                  className="h-5 w-5 text-[#50fa7b]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                  />
                </svg>
                <div>
                  <div className="font-semibold">Export Progress</div>
                  <div className="text-xs text-[#6272a4]">
                    Download backup file
                  </div>
                </div>
              </button>

              {/* Import Button */}
              <button
                onClick={handleImportClick}
                disabled={isImporting}
                className="flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm font-medium text-[#f8f8f2] transition-colors hover:bg-[#44475a] disabled:opacity-50"
              >
                <svg
                  className="h-5 w-5 text-[#8be9fd]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                  />
                </svg>
                <div>
                  <div className="font-semibold">
                    {isImporting ? 'Importing...' : 'Import Progress'}
                  </div>
                  <div className="text-xs text-[#6272a4]">
                    Restore from backup
                  </div>
                </div>
              </button>

              {/* Force Sync Button */}
              <button
                onClick={handleForceSync}
                disabled={isSyncing}
                className="flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-sm font-medium text-[#f8f8f2] transition-colors hover:bg-[#44475a] disabled:opacity-50"
              >
                <svg
                  className="h-5 w-5 text-[#f1fa8c]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                <div>
                  <div className="font-semibold">
                    {isSyncing ? 'Syncing...' : 'Force Sync'}
                  </div>
                  <div className="text-xs text-[#6272a4]">
                    Sync to database now
                  </div>
                </div>
              </button>

              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleFileChange}
                className="hidden"
              />

              {/* Info */}
              <div className="mt-2 border-t border-[#44475a] pt-2">
                <p className="px-4 py-2 text-xs text-[#6272a4]">
                  💡 Tip: Export regularly to avoid losing progress if browser
                  data is cleared.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Success/Error Message */}
      {message && (
        <div
          className={`fixed right-4 bottom-4 z-50 rounded-lg border-2 p-4 shadow-xl ${message.type === 'success'
              ? 'border-[#50fa7b] bg-[#50fa7b]/10 text-[#50fa7b]'
              : 'border-[#ff5555] bg-[#ff5555]/10 text-[#ff5555]'
            }`}
        >
          <div className="flex items-center gap-2">
            {message.type === 'success' ? (
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24">
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            ) : (
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24">
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            )}
            <span className="font-semibold">{message.text}</span>
          </div>
        </div>
      )}
    </div>
  );
}
