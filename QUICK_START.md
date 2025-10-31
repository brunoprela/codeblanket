# Quick Start Guide: Stack Auth + Neon Database

Get your CodeBlanket app running with authentication and cloud storage in 15 minutes.

## Prerequisites Checklist

- [ ] Node.js installed
- [ ] CodeBlanket frontend cloned
- [ ] Text editor open

## Step 1: Install Dependencies (Already Done ✓)

The required packages are already installed:

- `@stackframe/stack` - Authentication
- `@neondatabase/serverless` - Database driver

## Step 2: Create Stack Auth Account (5 minutes)

1. Go to https://app.stack-auth.com
2. Sign up for a free account
3. Create a new project (name it "CodeBlanket")
4. Copy these values from the dashboard:
   - Project ID (starts with `stack_`)
   - Publishable Client Key (starts with `pk_`)
   - Secret Server Key (starts with `sk_`)

## Step 3: Create Neon Database (5 minutes)

1. Go to https://console.neon.tech
2. Sign up for a free account
3. Create a new project (name it "codeblanket")
4. Copy the connection string from the dashboard
   - It looks like: `postgresql://user:pass@ep-xxx-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require`
5. Go to SQL Editor in Neon console
6. Paste the contents of `lib/db/schema.sql` and run it
   - This creates the `user_progress` and `user_videos` tables

## Step 3.5: Set Up Vercel Blob Storage (3 minutes)

1. Go to https://vercel.com/dashboard
2. Sign up or sign in
3. Create a new project (or use existing)
4. Go to **Storage** tab
5. Click **Create Database** → Select **Blob**
6. Copy the `BLOB_READ_WRITE_TOKEN` from Settings → Tokens

## Step 4: Configure Environment (2 minutes)

1. Create `.env.local` in the `frontend` directory:

```bash
# Copy from your Stack Auth dashboard
NEXT_PUBLIC_STACK_PROJECT_ID=stack_123456789
NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY=pk_123456789
STACK_SECRET_SERVER_KEY=sk_123456789

# Copy from your Neon dashboard
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require

# Copy from your Vercel Blob dashboard
BLOB_READ_WRITE_TOKEN=vercel_blob_rw_xxxxxxxxxxxxx
```

2. Save the file

## Step 5: Start the App (1 minute)

```bash
cd frontend
npm run dev
```

Visit http://localhost:3000

## Step 6: Test It Works (2 minutes)

### Test Anonymous Flow:

1. Open http://localhost:3000 in **Incognito/Private window**
2. Complete a coding problem
3. Close the tab and reopen - progress should persist ✓
4. Check DevTools > Application > IndexedDB > CodeBlanketDB ✓

### Test Authenticated Flow:

1. Open http://localhost:3000 in **normal window**
2. Click "Sign In" button in top-right
3. Create a new account (use any email/password)
4. You should see a migration dialog ✓
5. Click "Sync My Data" ✓
6. Complete a coding problem
7. Verify data in Neon:
   - Go to Neon console > SQL Editor
   - Run: `SELECT * FROM user_progress;`
   - You should see your data ✓

### Test Cross-Device Sync:

1. Sign out
2. Sign in from a different browser (or incognito)
3. Your progress should be there ✓

## Troubleshooting

### "Unauthorized" errors in console

- Check that all env variables are set correctly
- Restart dev server (`npm run dev`)

### Migration dialog doesn't appear

- Make sure you have local data (complete a problem first as anonymous)
- Check browser console for errors
- Clear `migration-completed-*` from localStorage to retry

### Database connection errors

- Verify DATABASE_URL is correct (copy-paste from Neon)
- Check that schema was created successfully
- Ensure `?sslmode=require` is in the connection string

### Stack Auth not loading

- Check that all Stack Auth env vars start with correct prefixes
- Verify Project ID in Stack Auth dashboard matches your `.env.local`
- Clear browser cache and cookies

## What Just Happened?

You now have:

- ✅ Working authentication with Stack Auth
- ✅ Cloud database with Neon PostgreSQL
- ✅ Video storage with Vercel Blob Storage
- ✅ Dual storage (IndexedDB for anonymous, PostgreSQL for authenticated)
- ✅ Automatic data migration when users sign up
- ✅ Cross-device sync for authenticated users

## Next Steps

- [ ] Customize the UI (components/AuthButtons.tsx, components/DataMigration.tsx)
- [ ] Add more auth methods in Stack Auth dashboard (Google, GitHub, etc.)
- [ ] Set up production deployment (Vercel, Netlify, etc.)
- [ ] Monitor usage in Stack Auth and Neon dashboards
- [ ] Consider upgrading to paid plans as you grow

## Production Deployment

When deploying:

1. **Add environment variables** to your hosting platform
2. **Update Stack Auth** allowed domains:
   - Stack Auth dashboard > Settings > Domains
   - Add your production domain (e.g., `https://codeblanket.com`)
3. **Test thoroughly** in production before announcing

## Support

- Stack Auth docs: https://docs.stack-auth.com
- Neon docs: https://neon.tech/docs
- This project: See `SETUP_AUTH_DATABASE.md` for detailed guide

## Cost Estimate

### Free Tier (Perfect for getting started):

- **Stack Auth**: 1,000 monthly active users
- **Neon**: 0.5 GB storage, ~200 hours compute/month
- **Cost**: $0/month

### When You Outgrow Free Tier:

- **Stack Auth Pro**: $50/month (10,000 MAU)
- **Neon Launch**: $19/month (3 GB storage)
- **Total**: ~$70/month for a growing app

You're all set! Happy coding! 🚀
