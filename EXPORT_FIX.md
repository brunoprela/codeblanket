# Export/Import Fix & Usage Guide

## What Was Wrong

The export was only showing module completion data because:

1. **Wrong key prefixes in migration** - The migration was looking for `'completed-problems'` but the actual localStorage key is `'codeblanket_completed_problems'`
2. **Data not synced to IndexedDB** - Only data that was migrated or explicitly synced would appear in exports

## What Was Fixed

### 1. **Fixed Migration Prefixes**

```typescript
// OLD (wrong)
const keysToMigrate = ['completed-problems', 'user-code-', 'module-'];

// NEW (correct)
const keysToMigrate = [
  'codeblanket_completed_problems',
  'codeblanket_code_',
  'module-',
  'codeblanket-',
];
```

### 2. **Added localStorage Fallback to Export**

Now when you export, it will:

- Try to get data from IndexedDB first
- If IndexedDB has minimal data, it will also include localStorage data
- Merge both sources (preferring IndexedDB)

### 3. **Added Force Sync Button**

A new "Force Sync" button in the Backup menu that:

- Manually syncs all localStorage data to IndexedDB
- Useful when you want to ensure everything is backed up before export

### 4. **Improved Import**

Import now restores data to BOTH:

- IndexedDB (for persistence)
- localStorage (for immediate access)

## How to Use

### Fresh Start / First Time

1. **Open the app** - The migration will run automatically
2. **Click "Backup" → "Force Sync"** - This ensures all your current data is synced
3. **Click "Backup" → "Export Progress"** - Download your backup file
4. **Check the exported JSON** - You should now see all your data

### Going Forward

1. **Complete problems as usual** - They auto-sync in the background
2. **Export regularly** (weekly recommended)
3. **Keep your backup files safe** (Google Drive, Dropbox, etc.)

### If Export Still Looks Empty

**Option 1: Force Sync**

1. Click "Backup" button in navbar
2. Click "Force Sync"
3. Wait for success message
4. Click "Export Progress"

**Option 2: Check Browser DevTools**

1. Open DevTools (F12)
2. Go to Application tab
3. Check localStorage - Look for keys starting with `codeblanket_`
4. Check IndexedDB - Look for `CodeBlanketDB` database

**Option 3: Manual Check**
Open browser console and run:

```javascript
// Check what's in localStorage
for (let i = 0; i < localStorage.length; i++) {
  const key = localStorage.key(i);
  if (key?.startsWith('codeblanket')) {
    console.log(key, localStorage.getItem(key));
  }
}
```

## What Gets Exported

Your export file should contain:

```json
{
  "version": "1.0",
  "exportDate": "2025-10-13T...",
  "data": {
    "codeblanket_completed_problems": ["binary-search", "two-sum", ...],
    "codeblanket_code_binary-search": "def binary_search...",
    "codeblanket_code_two-sum": "def two_sum...",
    "module-binary-search": { "section1": true, "section2": false },
    "module-arrays-hashing": { "introduction": true, ... }
  }
}
```

### Expected Keys:

- `codeblanket_completed_problems` - Array of completed problem IDs
- `codeblanket_code_*` - Saved code for each problem
- `module-*` - Module section completion status

## Troubleshooting

### "Still only seeing module data"

**Cause**: You might not have completed any problems or saved any code yet.

**Solution**:

1. Complete at least one problem (run all tests successfully)
2. Save some code in the editor
3. Force Sync
4. Export again

### "Export button does nothing"

**Cause**: JavaScript error or browser blocking download.

**Solution**:

1. Check browser console for errors
2. Try a different browser
3. Disable download blockers/extensions temporarily

### "Import says 'Invalid format'"

**Cause**: The JSON file might be corrupted or incomplete.

**Solution**:

1. Open the JSON file in a text editor
2. Verify it has `version`, `exportDate`, and `data` fields
3. Ensure JSON is valid (use jsonlint.com)

## Testing the Fix

To verify everything is working:

### Step 1: Complete a Problem

1. Go to any problem
2. Write a solution
3. Click "Run Tests"
4. Ensure all tests pass

### Step 2: Force Sync

1. Click "Backup" → "Force Sync"
2. Wait for "All data synced to IndexedDB successfully!" message

### Step 3: Export

1. Click "Backup" → "Export Progress"
2. Open the downloaded JSON file
3. Verify you see:
   - `codeblanket_completed_problems` with at least one problem ID
   - `codeblanket_code_*` with your saved code
   - `module-*` with any completed sections

### Step 4: Test Import (Optional)

1. Clear all browser data (Application → Clear storage)
2. Refresh the page
3. Click "Backup" → "Import Progress"
4. Select your exported JSON file
5. Verify all your progress is restored

## Additional Features

### Auto-Backup

The app automatically creates backups:

- Every time you load the page
- Every 5 minutes while the app is open
- Stored in localStorage as `codeblanket-auto-backup`

To access auto-backup:

```javascript
// In browser console
const backup = JSON.parse(localStorage.getItem('codeblanket-auto-backup'));
console.log(backup);
```

### Manual Recovery

If you lose your progress and have no export file:

1. Open DevTools → Console
2. Run: `localStorage.getItem('codeblanket-auto-backup')`
3. Copy the entire JSON string
4. Create a new file `recovery.json` and paste:

```json
{
  "version": "1.0",
  "exportDate": "2025-10-13T00:00:00.000Z",
  "data": {
    // paste the backup data here
  }
}
```

5. Import this file

## Questions?

See `STORAGE.md` for detailed documentation on the storage system.
