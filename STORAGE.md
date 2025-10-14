# Storage System Documentation

## Overview

CodeBlanket uses a **dual-layer storage system** to ensure your progress is never lost:

1. **localStorage** - Fast cache for immediate read/write
2. **IndexedDB** - Persistent database that survives browser cleanups
3. **Export/Import** - Manual backup files you control

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Actions                       │
│  (Complete Problem, Save Code, Mark Section Done)   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│              storage.ts (Helpers)                    │
│  • getCompletedProblems()                           │
│  • markProblemCompleted()                           │
│  • saveUserCode()                                   │
└───────────┬─────────────────────────┬───────────────┘
            │                         │
    ┌───────▼──────┐         ┌────────▼────────┐
    │ localStorage │         │   IndexedDB     │
    │  (Cache)     │◄────────┤  (Persistent)   │
    └──────────────┘  Sync   └─────────────────┘
            │                         │
            └─────────┬───────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │  Export/Import   │
            │  (JSON Files)    │
            └──────────────────┘
```

## Files

### Core Storage

- **`lib/helpers/storage.ts`** - Main storage API (synchronous)
  - `getCompletedProblems()` - Get all completed problem IDs
  - `markProblemCompleted(id)` - Mark a problem as completed
  - `markProblemIncomplete(id)` - Unmark a problem
  - `saveUserCode(id, code)` - Save user's code for a problem
  - `getUserCode(id)` - Retrieve saved code
  - `clearUserCode(id)` - Delete saved code
  - `saveCustomTestCases(id, tests)` - Save custom test cases
  - `getCustomTestCases(id)` - Retrieve custom test cases
  - `clearCustomTestCases(id)` - Delete custom test cases
  - `saveMultipleChoiceProgress(moduleId, sectionId, completedIds)` - Save MC quiz progress
  - `getMultipleChoiceProgress(moduleId, sectionId)` - Get MC quiz progress
  - `clearMultipleChoiceProgress(moduleId, sectionId)` - Clear MC quiz progress
  - `saveCompletedSections(moduleId, sectionIds)` - Save module section completion
  - `getCompletedSections(moduleId)` - Get completed sections
  - `markSectionCompleted(moduleId, sectionId)` - Mark section as completed
  - `markSectionIncomplete(moduleId, sectionId)` - Mark section as incomplete
  - `isSectionCompleted(moduleId, sectionId)` - Check if section is completed

- **`lib/helpers/indexeddb.ts`** - IndexedDB wrapper (asynchronous)
  - `setItem(key, value)` - Store data in IndexedDB
  - `getItem(key)` - Retrieve data from IndexedDB
  - `removeItem(key)` - Delete data from IndexedDB
  - `getAllData()` - Export all data
  - `importData(data)` - Import all data
  - `migrateFromLocalStorage()` - One-time migration
  - `saveVideo(videoId, blob)` - Save video recording to IndexedDB
  - `getVideosForQuestion(questionIdPrefix)` - Get all videos for a question
  - `deleteVideo(videoId)` - Delete a specific video
  - `getCompletedDiscussionQuestionsCount()` - Count unique questions with videos

- **`lib/helpers/export-import.ts`** - Export/Import functionality
  - `exportProgress()` - Download progress as JSON file
  - `importProgress(file)` - Restore progress from JSON file
  - `createAutoBackup()` - Create auto-backup in localStorage

### UI Components

- **`components/ExportImportMenu.tsx`** - Export/Import dropdown in navbar
- **`components/StorageInitializer.tsx`** - Initializes storage on app load
- **`lib/hooks/useStorageInit.ts`** - Hook for storage initialization

## How It Works

### 1. Dual-Layer Storage

When you complete a problem or save code:

```typescript
// User completes a problem
markProblemCompleted('binary-search');

// What happens:
// 1. Writes to localStorage immediately (fast!)
localStorage.setItem('codeblanket_completed_problems', [...]);

// 2. Syncs to IndexedDB in background (persistent!)
setItem('codeblanket_completed_problems', [...]);
```

### 2. Automatic Migration

On first app load, existing localStorage data is automatically migrated to IndexedDB:

```typescript
// Runs once per browser
useStorageInit() → migrateFromLocalStorage()
```

### 3. Auto-Backups

The app creates auto-backups:

- **On every page load**
- **Every 5 minutes** while the app is open
- Stored in localStorage as `codeblanket-auto-backup`

### 4. Manual Export/Import

#### To Export:

1. Click "Backup" button in navbar
2. Click "Export Progress"
3. File downloads: `codeblanket-progress-YYYY-MM-DD.json`

#### To Import:

1. Click "Backup" button in navbar
2. Click "Import Progress"
3. Select your JSON file
4. App automatically refreshes with restored data

## Data Stored

### Completed Problems

```json
{
  "codeblanket_completed_problems": [
    "binary-search",
    "two-sum",
    "valid-anagram"
  ]
}
```

### User Code

```json
{
  "codeblanket_code_binary-search": "def binary_search(nums, target):\n    # user code...",
  "codeblanket_code_two-sum": "def two_sum(nums, target):\n    # user code..."
}
```

### Custom Test Cases

```json
{
  "codeblanket_tests_binary-search": [
    { "input": "[1, 2, 3], 2", "expected": "1" },
    { "input": "[1, 2, 3], 4", "expected": "-1" }
  ]
}
```

### Multiple Choice Quiz Progress

```json
{
  "mc-quiz-python-fundamentals-variables": ["question-1", "question-3"]
}
```

### Module Section Completion

```json
{
  "module-binary-search-completed": ["section-1", "section-2"]
}
```

### Video Recordings (Discussion Questions)

Stored in IndexedDB video store as Blobs:

- Video ID format: `{moduleId}-{sectionId}-{questionId}-{timestamp}`
- Exported as base64-encoded strings in export file

## Export File Format

```json
{
  "version": "1.0",
  "exportDate": "2024-10-13T12:00:00.000Z",
  "data": {
    "codeblanket_completed_problems": ["binary-search", "two-sum"],
    "codeblanket_code_binary-search": "def binary_search(nums, target):\n    ...",
    "codeblanket_tests_binary-search": [{ "input": "...", "expected": "..." }],
    "mc-quiz-python-fundamentals-variables": ["question-1", "question-2"],
    "module-binary-search-completed": ["section-1"]
  },
  "videos": [
    {
      "id": "python-fundamentals-variables-question1-1697203200000",
      "data": "base64-encoded-video-data...",
      "timestamp": 1697203200000
    }
  ]
}
```

**Note:** Videos are stored as base64-encoded strings, which can make the export file large (potentially 10-100+ MB depending on number and length of videos). The export function will log the total size of videos being exported.

## Why This Approach?

### ✅ Advantages

1. **No Backend Needed** - Everything runs in the browser
2. **Fast Performance** - localStorage cache for instant access
3. **Persistent Data** - IndexedDB survives browser cleanups better than localStorage
4. **User Control** - Export/import files give users full control
5. **Portable** - Take your progress anywhere via JSON files
6. **Privacy** - All data stays on your device

### ⚠️ Limitations

1. **Not synchronized across devices** - Each device has its own data
2. **Manual export required** for backups
3. **Browser-dependent** - Clearing browser data will still delete everything
4. **No collaboration** - Can't share progress with others in real-time

## Best Practices

### For Users

1. **Export regularly** (weekly or after major progress)
2. **Keep backup files safe** (Google Drive, Dropbox, etc.)
3. **Before clearing browser data**, export your progress first
4. **Name your exports** meaningfully, e.g., `codeblanket-progress-completed-arrays.json`

### For Developers

```typescript
// Always use the storage helpers, never access localStorage directly
import { markProblemCompleted } from '@/lib/helpers/storage';

// Good ✅
markProblemCompleted(problemId);

// Bad ❌
localStorage.setItem('completed', problemId);
```

## Future Enhancements

Potential upgrades (if needed):

1. **Cloud Sync** - Add Firebase/Supabase for cross-device sync
2. **GitHub Integration** - Store progress in private GitHub Gist
3. **Auto-Upload** - Periodic uploads to cloud storage
4. **Conflict Resolution** - Merge progress from multiple devices
5. **Progress Analytics** - Track learning velocity, streaks, etc.

## Troubleshooting

### "My progress disappeared!"

1. Check if you have an auto-backup:
   - Open browser DevTools → Console
   - Type: `localStorage.getItem('codeblanket-auto-backup')`
   - If data exists, it can be manually restored

2. Check if you have an exported file
   - Look in your Downloads folder for `codeblanket-progress-*.json`

### "Import isn't working"

- Ensure the file is valid JSON
- Check the file wasn't corrupted
- Make sure it has the correct format (version, exportDate, data fields)

### "I want to start fresh"

1. Export your current progress first (just in case!)
2. Open DevTools → Application → Clear storage
3. Refresh the page

## Testing

To test the system:

```javascript
// In browser DevTools console

// Test export
window.dispatchEvent(new CustomEvent('test-export'));

// Test storage
localStorage.setItem('codeblanket_completed_problems', '["test-problem"]');

// Check IndexedDB
// DevTools → Application → IndexedDB → CodeBlanketDB
```

---

**Questions?** Open an issue or check the code in `lib/helpers/`.
