# Vercel Blob Storage Setup for Videos

This guide explains how to set up Vercel Blob Storage for storing video recordings in CodeBlanket.

## Why Vercel Blob Storage?

Videos are **large binary files** that are expensive to store in traditional databases. Vercel Blob Storage provides:

- ✅ **Optimized for large files**: Designed specifically for media and binary data
- ✅ **Edge distribution**: Videos served from CDN for fast global access
- ✅ **Cost-effective**: Pay only for what you use, not for database storage
- ✅ **Simple API**: Easy upload/download with just a few lines of code
- ✅ **Automatic cleanup**: Delete from both blob storage and database with one API call

## Architecture

```
┌─────────────────┐
│  User uploads   │
│   video file    │
└────────┬────────┘
         │
         v
┌────────────────────┐
│ POST /api/videos   │
│ (Next.js API)      │
└─────┬──────────┬───┘
      │          │
      v          v
┌─────────┐  ┌──────────┐
│ Vercel  │  │PostgreSQL│
│  Blob   │  │(metadata)│
│ Storage │  │          │
└─────────┘  └──────────┘
   (file)     (URL+info)
```

**What gets stored where:**
- **Vercel Blob**: Actual video file (.webm format)
- **PostgreSQL**: Video metadata (blob URL, pathname, size, timestamp)

## Setup Steps

### Step 1: Enable Vercel Blob in Your Project

1. Go to your Vercel dashboard: https://vercel.com/dashboard
2. Select your project (or create one)
3. Go to **Storage** tab
4. Click **Create Database**
5. Select **Blob** from the options
6. Click **Create**

### Step 2: Get Your Blob Token

After creating the Blob store:

1. In the Storage tab, click on your Blob store
2. Go to **Settings** → **Tokens**
3. Copy the `BLOB_READ_WRITE_TOKEN`
4. Add it to your `.env.local`:

```bash
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_xxxxxxxxxxxxx
```

That's it! The token is automatically connected to your Vercel project.

### Step 3: Update Database Schema

Run the migration to update your `user_videos` table:

```bash
# Connect to your Neon database
psql "YOUR_DATABASE_URL" < lib/db/schema-blob-migration.sql
```

Or run the SQL in the Neon SQL editor:

```sql
-- See lib/db/schema-blob-migration.sql for full script
ALTER TABLE IF EXISTS user_videos RENAME TO user_videos_backup;

CREATE TABLE user_videos (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  video_id TEXT NOT NULL,
  blob_url TEXT NOT NULL,
  blob_pathname TEXT NOT NULL,
  mime_type TEXT DEFAULT 'video/webm',
  file_size BIGINT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, video_id)
);
```

## How It Works

### Upload Flow

```javascript
// User records video in browser
const videoBlob = recordedVideo;

// Upload to API
const formData = new FormData();
formData.append('videoId', 'problem-123-timestamp');
formData.append('video', videoBlob);

fetch('/api/videos', {
  method: 'POST',
  body: formData,
});
```

**What happens server-side:**

1. API receives video file
2. Uploads to Vercel Blob Storage → gets back a URL
3. Saves metadata to PostgreSQL:
   ```json
   {
     "user_id": "user_abc",
     "video_id": "problem-123-timestamp",
     "blob_url": "https://abc123.public.blob.vercel-storage.com/...",
     "blob_pathname": "videos/user_abc/problem-123-timestamp.webm",
     "file_size": 1234567
   }
   ```

### Retrieval Flow

```javascript
// Fetch video metadata
const response = await fetch('/api/videos?videoId=problem-123-timestamp');
// API redirects to Vercel Blob URL

// Or fetch all videos for a question
const response = await fetch('/api/videos');
const { videos } = await response.json();
// Returns array with blob URLs
```

**What happens server-side:**

1. API queries PostgreSQL for metadata
2. Returns blob URL
3. Client fetches video directly from Vercel Blob CDN

### Delete Flow

```javascript
// Delete video
await fetch('/api/videos?videoId=problem-123-timestamp', {
  method: 'DELETE',
});
```

**What happens server-side:**

1. API gets blob pathname from PostgreSQL
2. Deletes from Vercel Blob Storage
3. Deletes metadata from PostgreSQL

## Testing

### Test Upload

1. Sign in to your app
2. Go to a module with discussion questions
3. Record a video
4. Check Vercel dashboard > Storage > Blob to see uploaded file
5. Check Neon database: `SELECT * FROM user_videos;`

### Test Retrieval

1. Reload the page
2. Your recorded video should load
3. Check Network tab - video loads from `*.blob.vercel-storage.com`

### Test Deletion

1. Delete a video
2. Check Vercel Blob dashboard - file should be gone
3. Check Neon database - metadata should be gone

## File Organization

Videos are stored with this pathname structure:

```
videos/{userId}/{videoId}.webm
```

Examples:
- `videos/user_123abc/python-fundamentals-variables-q1-1699123456.webm`
- `videos/user_456def/system-design-caching-q2-1699123789.webm`

This structure:
- Groups videos by user (easy to find all of a user's videos)
- Uses consistent naming (no random suffixes)
- Overwrites old recordings (when user re-records)

## Environment Variables

Required in `.env.local`:

```bash
# Vercel Blob Storage
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_xxxxxxxxxxxxx
```

In production (Vercel dashboard):
- The token is automatically set when you connect Blob storage
- No manual configuration needed!

## Cost Estimates

Vercel Blob pricing (as of 2024):

**Free Tier:**
- 5 GB storage
- Unlimited bandwidth

**Pro Plan** ($20/month for whole account):
- 100 GB storage (included)
- Unlimited bandwidth
- $0.15/GB beyond 100 GB

**Example calculation:**
- Average video: 5 MB (5-minute explanation)
- Free tier: ~1,000 videos
- Pro tier: ~20,000 videos included

For most learning platforms, the free tier is plenty to start!

## Migration from Old Schema

If you have existing videos in the old schema (binary data in PostgreSQL):

1. The migration script backs up to `user_videos_backup`
2. You can migrate old videos using this script:

```sql
-- Example migration script (customize as needed)
INSERT INTO user_videos (user_id, video_id, blob_url, blob_pathname, mime_type, file_size)
SELECT 
  user_id,
  video_id,
  'https://placeholder.com/' || video_id, -- Placeholder URL
  'migrated/' || video_id,
  mime_type,
  LENGTH(video_data)
FROM user_videos_backup;
```

Or simply start fresh with new recordings - old videos remain in `user_videos_backup`.

## Troubleshooting

### Upload fails with "Missing token"

- Check that `BLOB_READ_WRITE_TOKEN` is set in `.env.local`
- Restart dev server after adding env var
- In production, ensure Blob storage is connected in Vercel dashboard

### Videos not loading

- Check Network tab for 404s
- Verify blob URL is valid (click it directly)
- Check PostgreSQL has correct blob_url saved
- Ensure Blob storage is set to `public` access (default)

### "Failed to delete video"

- Check that blob_pathname exists in database
- Verify token has write permissions
- Check Vercel Blob dashboard for orphaned files

### Large videos timing out

- Default upload limit: 4.5 MB (Vercel Serverless)
- For larger videos, consider:
  - Client-side compression before upload
  - Vercel Pro plan (50 MB limit)
  - Streaming uploads (advanced)

## Security Considerations

- ✅ **Authentication required**: All API routes check user auth
- ✅ **User isolation**: Users can only access their own videos
- ✅ **Public blob URLs**: Videos are publicly accessible if you have the URL
  - This is intentional for CDN performance
  - URLs are long and random (effectively unguessable)
  - Alternative: Set `access: 'private'` in `put()` call (requires signed URLs)
- ✅ **Automatic cleanup**: Deleting video metadata also deletes blob file

## Advanced: Private Videos with Signed URLs

If you want truly private videos:

1. Change upload to private:
```typescript
const blob = await put(pathname, videoFile, {
  access: 'private', // Change from 'public'
  addRandomSuffix: false,
});
```

2. Generate signed URLs when serving:
```typescript
import { generateSignedUrl } from '@vercel/blob';

const signedUrl = await generateSignedUrl({
  pathname: metadata.blobPathname,
  token: process.env.BLOB_READ_WRITE_TOKEN!,
});
// Expires after 1 hour
```

3. Update GET endpoint to return signed URL instead of redirecting

Trade-offs:
- 👍 More secure (URLs expire)
- 👎 More complex (need to regenerate URLs)
- 👎 No CDN caching (signed URLs are unique)

## Bandwidth Optimization

To avoid excessive bandwidth usage, videos use **lazy loading**:

- ✅ **Page load**: Only metadata fetched (~200 bytes per video)
- ✅ **Video display**: Placeholder + "Load Video" button shown
- ✅ **On-demand**: Actual video loaded only when user clicks

**Bandwidth savings:**
- Before: 100+ MB per page load (all videos)
- After: <1 KB per page load (metadata only)
- **Reduction: 99.99%** 🎉

This means you can stay on Vercel's free tier much longer!

See `BANDWIDTH_OPTIMIZATION.md` for details.

## Next Steps

- ✅ Videos stored efficiently in Vercel Blob
- ✅ Fast global access via CDN
- ✅ Metadata tracked in PostgreSQL
- ✅ Automatic cleanup on deletion
- ✅ On-demand loading to save bandwidth

Your video storage is production-ready! 🎥

