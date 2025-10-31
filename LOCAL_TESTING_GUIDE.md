# Local Testing Guide

## Prerequisites Checklist

✅ Stack Auth credentials in `.env`
✅ Neon Database URL in `.env`
⏳ Vercel Blob token (we'll add this)
⏳ Database schema created (we'll do this)

## Step 1: Set Up Vercel Blob Storage (5 minutes)

### Option A: Create Vercel Blob Store

1. Go to https://vercel.com/dashboard
2. Create a new project (or use existing)
3. Go to **Storage** tab
4. Click **Create Database** → Select **Blob**
5. After creation, go to **Settings** → **Tokens**
6. Copy the `BLOB_READ_WRITE_TOKEN`
7. Add to your `.env`:

```bash
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_xxxxxxxxxxxxx
```

### Option B: Test Without Videos First

If you want to test authentication and progress storage first without videos:
- Skip Vercel Blob for now
- Videos will fallback to IndexedDB (still works!)
- Add Vercel Blob later when ready

## Step 2: Create Database Tables (2 minutes)

Run the schema SQL in your Neon database:

### Method 1: Using psql (Command Line)

```bash
# From the frontend directory
psql "postgresql://neondb_owner:npg_ztd3cDkr4bSw@ep-muddy-dream-a4w0jsgc-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require" < lib/db/schema.sql
```

### Method 2: Using Neon Console (Web UI)

1. Go to https://console.neon.tech
2. Select your project
3. Go to **SQL Editor**
4. Open `lib/db/schema.sql` in your code editor
5. Copy the entire contents
6. Paste into Neon SQL Editor
7. Click **Run**

You should see:
```
CREATE TABLE
CREATE INDEX
CREATE FUNCTION
CREATE TRIGGER
```

## Step 3: Start the Development Server

```bash
cd /Users/bruno/Developer/codeblanket/frontend
npm run dev
```

You should see:
```
▲ Next.js 15.x.x
- Local:        http://localhost:3000
✓ Ready in Xms
```

## Step 4: Test Anonymous User Flow (IndexedDB)

1. Open http://localhost:3000 in **Incognito/Private window**
2. **Test**: Complete a coding problem
   - Navigate to a problem
   - Write some code
   - Mark as complete
3. **Verify**: Close tab and reopen
   - Progress should persist ✓
4. **Check**: Open DevTools → Application → IndexedDB → CodeBlanketDB
   - You should see your data ✓

## Step 5: Test Authenticated User Flow (PostgreSQL)

### First Sign-Up

1. Open http://localhost:3000 in **normal window**
2. Click **"Sign In"** button (top right)
3. Stack Auth modal appears
4. Click **"Sign up"** tab
5. Enter email and password (any test email works)
6. Click **"Sign up"**

### Test Data Storage

1. **Complete a problem**:
   - Go to any coding problem
   - Write some code
   - Save it

2. **Verify in Database**:
   - Go to Neon Console → SQL Editor
   - Run: `SELECT * FROM user_progress;`
   - You should see your data! ✓

3. **Test Cross-Device Sync**:
   - Sign out
   - Sign in from different browser (or incognito)
   - Your progress should be there ✓

## Step 6: Test Data Migration

### If you have local data (IndexedDB):

1. **As anonymous**: Complete 2-3 problems
2. **Sign in**: Click "Sign In" button
3. **Migration dialog appears**: Shows "Sync My Data" option
4. **Click "Sync My Data"**
5. **Wait for success**: ✓ message appears
6. **Verify**: Check Neon database - your problems should be there!

### Query to verify migration:

```sql
-- See all your synced data
SELECT key, value, created_at 
FROM user_progress 
ORDER BY created_at DESC;
```

## Step 7: Test Video Recording (If Vercel Blob is set up)

1. Navigate to any module with discussion questions
2. Scroll to a discussion question
3. Click **"Start Camera"**
4. Allow camera permissions
5. Click **"Record"**
6. Speak for a few seconds
7. Click **"Stop & Save"**
8. Success message appears ✓

### Verify Video Storage:

**Vercel Blob Dashboard:**
- Go to Vercel → Storage → Blob
- You should see your video file
- Path: `videos/{userId}/{videoId}.webm`

**Neon Database:**
```sql
SELECT video_id, blob_url, file_size, created_at 
FROM user_videos 
ORDER BY created_at DESC;
```

## Step 8: Test On-Demand Video Loading (Bandwidth Optimization)

1. Refresh the module page
2. **Without clicking**: Videos show as placeholders ✓
3. **Network tab**: No large video downloads ✓
4. **Click "Load Video"**: 
   - Loading spinner appears
   - Video downloads and plays
5. **Bandwidth saved**: 99.99% reduction! ✓

## Troubleshooting

### "Unauthorized" Errors

**Symptom**: API calls return 401
**Fix**:
```bash
# Restart dev server after adding env vars
# Ctrl+C to stop, then:
npm run dev
```

### Database Connection Failed

**Symptom**: Can't connect to Neon
**Check**:
- Is DATABASE_URL correct in `.env`?
- Does it include `?sslmode=require`?
- Try connecting with psql to verify

### Stack Auth Not Loading

**Symptom**: Sign in button doesn't work
**Check**:
- All 3 Stack Auth env vars are set
- Restart dev server
- Clear browser cache/cookies
- Check Stack Auth dashboard is active

### Migration Dialog Not Appearing

**Symptom**: No migration prompt when signing in
**Steps**:
1. Make sure you have IndexedDB data first
2. Clear `migration-completed-*` from localStorage
3. Sign in again

### Videos Not Uploading

**Without Vercel Blob Token**:
- Videos save to IndexedDB (still works!)
- Add token later for Vercel Blob

**With Vercel Blob Token**:
- Check token is correct in `.env`
- Verify Vercel Blob store is created
- Check Network tab for error details

## Quick Verification Checklist

Run these checks to verify everything works:

### Authentication
- [ ] Sign up creates new account
- [ ] Sign in works with existing account
- [ ] User button shows in nav bar
- [ ] Sign out works

### Data Storage (Anonymous)
- [ ] Problems saved to IndexedDB
- [ ] Code persists across page reloads
- [ ] Quiz progress tracked

### Data Storage (Authenticated)
- [ ] Problems saved to PostgreSQL
- [ ] Data visible in Neon Console
- [ ] Cross-browser sync works

### Migration
- [ ] Migration dialog appears
- [ ] Data migrates successfully
- [ ] No duplicates in database

### Videos (if Blob set up)
- [ ] Can record videos
- [ ] Videos save to Vercel Blob
- [ ] Metadata in PostgreSQL
- [ ] On-demand loading works

## Common Commands

### Check Database Connection
```bash
psql "$DATABASE_URL" -c "SELECT version();"
```

### View All Tables
```bash
psql "$DATABASE_URL" -c "\dt"
```

### Clear Local Data (Start Fresh)
```javascript
// Run in browser console
localStorage.clear();
indexedDB.deleteDatabase('CodeBlanketDB');
location.reload();
```

### Check Environment Variables
```bash
# In frontend directory
cat .env
```

## Next Steps After Local Testing

Once everything works locally:

1. **Deploy to Vercel**:
   - Push code to GitHub
   - Connect to Vercel
   - Add env vars in Vercel dashboard
   - Deploy automatically connects Blob storage

2. **Update Stack Auth**:
   - Add production domain to allowed origins
   - Update redirect URLs

3. **Test in Production**:
   - Verify sign-up flow
   - Test data persistence
   - Check video uploads

## Performance Tips

### Development Mode
- Page loads slower than production
- Hot reload can cause state issues
- Clear cache if weird behavior

### Testing Multiple Users
- Use different browsers
- Or multiple incognito windows
- Or different browser profiles

### Database Queries
- Use Neon Console SQL Editor
- Add indexes if queries slow
- Monitor query performance

## Debugging Tools

### Browser DevTools
- **Console**: See errors and logs
- **Network**: Check API calls
- **Application**: View localStorage/IndexedDB
- **Sources**: Debug breakpoints

### Neon Console
- **SQL Editor**: Run queries
- **Monitoring**: Check connections
- **Logs**: See query logs

### Vercel Dashboard
- **Blob Storage**: See uploaded files
- **Bandwidth**: Monitor usage
- **Logs**: Check errors

## Success Criteria

You should be able to:
✓ Sign up and sign in
✓ Save progress as authenticated user
✓ See data in Neon database
✓ Test as anonymous user with IndexedDB
✓ Migrate data when signing in
✓ Record videos (if Blob set up)
✓ See bandwidth optimization working

If all of these work, you're ready for production! 🚀

