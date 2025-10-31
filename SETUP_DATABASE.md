# Quick Database Setup

You're seeing errors because the database tables need to be created first!

## Setup (2 minutes)

### Step 1: Go to Neon Console

Visit: https://console.neon.tech

### Step 2: Select Your Project

Click on the project that matches your DATABASE_URL

### Step 3: Open SQL Editor

Click **SQL Editor** in the left sidebar

### Step 4: Run the Schema

Copy and paste this entire SQL script into the editor:

```sql
-- CodeBlanket User Progress Database Schema
-- For use with Neon PostgreSQL

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User progress data table
-- Stores all user progress data (completed problems, code, quiz answers, etc.)
CREATE TABLE IF NOT EXISTS user_progress (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(user_id, key)
);

-- Index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_user_progress_user_id ON user_progress(user_id);

-- Index for faster queries by user_id and key pattern
CREATE INDEX IF NOT EXISTS idx_user_progress_user_key ON user_progress(user_id, key);

-- Video recordings table
-- Stores metadata for videos stored in Vercel Blob Storage
CREATE TABLE IF NOT EXISTS user_videos (
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

-- Index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_user_videos_user_id ON user_videos(user_id);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
DROP TRIGGER IF EXISTS update_user_progress_updated_at ON user_progress;
CREATE TRIGGER update_user_progress_updated_at
  BEFORE UPDATE ON user_progress
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_videos_updated_at ON user_videos;
CREATE TRIGGER update_user_videos_updated_at
  BEFORE UPDATE ON user_videos
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();
```

### Step 5: Click "Run"

You should see:
- ✓ CREATE EXTENSION
- ✓ CREATE TABLE (2 times)
- ✓ CREATE INDEX (3 times)
- ✓ CREATE FUNCTION
- ✓ CREATE TRIGGER (2 times)

### Step 6: Verify Tables Were Created

Run this query in the SQL Editor:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';
```

You should see:
- user_progress
- user_videos

## Done! 🎉

Now refresh your browser at http://localhost:3000

The error should be gone and your progress will save to the cloud!

## Quick Test

After refreshing:
1. Complete a coding problem
2. Go back to Neon SQL Editor
3. Run: `SELECT * FROM user_progress;`
4. You should see your data! 🎊

