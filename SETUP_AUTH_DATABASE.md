# Stack Auth + Neon PostgreSQL Setup Guide

This guide explains how to set up authentication with Stack Auth and database storage with Neon PostgreSQL for CodeBlanket.

## Overview

CodeBlanket now uses a **dual-storage strategy**:

- **Anonymous users**: Data stored in browser IndexedDB (client-side)
- **Authenticated users**: Data stored in Neon PostgreSQL (server-side)

This allows anyone to try the application without signing up, while giving authenticated users cloud-based persistence and cross-device sync.

## Architecture

```
┌─────────────────┐
│   User Action   │
│ (save progress) │
└────────┬────────┘
         │
         v
┌─────────────────────┐
│ Storage Adapter     │
│ (auto-detects auth) │
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    v         v
┌───────┐  ┌──────────┐
│IndexDB│  │PostgreSQL│
│(anon) │  │ (authed) │
└───────┘  └──────────┘
```

## Prerequisites

1. **Stack Auth Account**: Sign up at [stack-auth.com](https://stack-auth.com)
2. **Neon Database**: Sign up at [neon.tech](https://neon.tech)

## Step 1: Set Up Stack Auth

1. Go to [Stack Auth Dashboard](https://app.stack-auth.com)
2. Create a new project
3. Get your credentials from the dashboard:
   - `NEXT_PUBLIC_STACK_PROJECT_ID`
   - `NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY`
   - `STACK_SECRET_SERVER_KEY`

## Step 2: Set Up Neon Database

1. Go to [Neon Console](https://console.neon.tech)
2. Create a new project
3. Copy your connection string (it should look like: `postgresql://user:password@xxx.neon.tech/dbname?sslmode=require`)
4. Run the schema migration:

```bash
# Connect to your Neon database and run the schema
psql "YOUR_DATABASE_URL" < lib/db/schema.sql
```

Or manually execute the SQL in `lib/db/schema.sql` through the Neon SQL editor.

## Step 3: Configure Environment Variables

Create a `.env.local` file in the `frontend` directory:

```bash
# Stack Auth Configuration
NEXT_PUBLIC_STACK_PROJECT_ID=your_project_id_here
NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY=your_publishable_key_here
STACK_SECRET_SERVER_KEY=your_secret_key_here

# Neon Database
DATABASE_URL=postgresql://user:password@xxx.neon.tech/dbname?sslmode=require
```

## Step 4: Test the Setup

1. Start the development server:

```bash
npm run dev
```

2. Visit `http://localhost:3000`

3. Test anonymous usage:
   - Complete a problem without signing in
   - Your progress should be saved to IndexedDB
   - Check browser DevTools > Application > IndexedDB > CodeBlanketDB

4. Test authenticated usage:
   - Click "Sign In" button
   - Create an account or sign in
   - You should see a migration dialog offering to sync your local data
   - Click "Sync My Data"
   - Your progress should now be in PostgreSQL
   - Verify by checking the Neon console:
     ```sql
     SELECT * FROM user_progress;
     SELECT * FROM user_videos;
     ```

## How It Works

### Storage Adapter

All storage operations go through `lib/helpers/storage-adapter.ts`, which:

1. Checks if the user is authenticated (via `/api/auth/check`)
2. Routes to the appropriate backend:
   - **IndexedDB**: For anonymous users (via direct IndexedDB calls)
   - **PostgreSQL**: For authenticated users (via API routes)
3. Includes automatic fallback to IndexedDB if PostgreSQL fails

### API Routes

- `GET/POST/DELETE /api/progress` - User progress data
- `POST /api/progress/import` - Bulk import for migration
- `GET/POST/DELETE /api/videos` - Video recordings
- `GET /api/auth/check` - Check authentication status

### Database Schema

Two main tables in PostgreSQL:

1. **user_progress**: Stores all progress data (completed problems, code, quiz answers)

   ```sql
   CREATE TABLE user_progress (
     id UUID PRIMARY KEY,
     user_id TEXT NOT NULL,
     key TEXT NOT NULL,
     value JSONB NOT NULL,
     created_at TIMESTAMPTZ,
     updated_at TIMESTAMPTZ,
     UNIQUE(user_id, key)
   );
   ```

2. **user_videos**: Stores video recordings
   ```sql
   CREATE TABLE user_videos (
     id UUID PRIMARY KEY,
     user_id TEXT NOT NULL,
     video_id TEXT NOT NULL,
     video_data BYTEA NOT NULL,
     mime_type TEXT,
     created_at TIMESTAMPTZ,
     updated_at TIMESTAMPTZ,
     UNIQUE(user_id, video_id)
   );
   ```

### Data Migration

When a user signs in for the first time, the `DataMigration` component:

1. Detects existing IndexedDB data
2. Shows a dialog offering to sync it
3. Migrates all data to PostgreSQL via the import API
4. Marks migration as complete (stored in localStorage to prevent re-prompting)

Users can:

- **Sync My Data**: Upload all local data to cloud
- **Skip**: Continue with empty cloud storage (local data remains in IndexedDB)

## Testing

### Test Anonymous Usage

```bash
# Open in incognito/private window
# Complete some problems
# Close browser
# Reopen - progress should persist (IndexedDB)
```

### Test Authenticated Usage

```bash
# Sign in
# Complete some problems
# Sign out and sign in from different browser
# Progress should sync (PostgreSQL)
```

### Test Migration

```bash
# In incognito: Complete problems (creates IndexedDB data)
# Sign up/sign in
# Migration dialog should appear
# Click "Sync My Data"
# Verify data in Neon console
```

## Troubleshooting

### Migration Dialog Not Appearing

- Check browser console for errors
- Verify IndexedDB has data: DevTools > Application > IndexedDB
- Clear `migration-completed-*` from localStorage to retry

### Data Not Saving to PostgreSQL

- Check browser console for API errors
- Verify DATABASE_URL is correct
- Check Neon console for connection issues
- Verify schema was created: `SELECT * FROM user_progress LIMIT 1;`

### Authentication Not Working

- Verify Stack Auth credentials in `.env.local`
- Check Stack Auth dashboard for project status
- Clear cookies and try again
- Check browser console for Stack Auth errors

## Production Deployment

When deploying to production:

1. Add environment variables to your hosting platform (Vercel, Netlify, etc.)
2. Ensure DATABASE_URL uses SSL (Neon includes `?sslmode=require`)
3. Update Stack Auth allowed origins in dashboard
4. Test sign-in flow in production
5. Monitor Neon database usage and upgrade plan if needed

## Security Considerations

- All API routes check authentication before allowing access
- Users can only access their own data (enforced by `user_id` in queries)
- Database credentials are server-side only (never exposed to client)
- Stack Auth handles password security and session management
- Neon uses SSL by default for all connections

## Cost Considerations

### Stack Auth

- Free tier: 1,000 MAUs (Monthly Active Users)
- Paid plans start at $50/month for more users

### Neon

- Free tier: 0.5 GB storage, 191.9 hours compute per month
- Paid plans start at $19/month for more storage/compute

Both services are generous for small to medium applications.

## Future Enhancements

Possible improvements:

- Real-time sync across devices using WebSockets
- Offline-first with background sync queue
- Data export to other formats (PDF, JSON)
- Social features (leaderboards, sharing progress)
- Team/classroom accounts with admin dashboards
