# 🚨 FIX: Neon "Blocked Network" Error

## The Problem

You're seeing:

```
"This connection is trying to access this endpoint from a blocked network."
```

This means **Neon has IP restrictions enabled** and is blocking:

- ❌ Your localhost (development)
- ❌ Vercel's serverless functions (production)

## The Solution (2 minutes)

### Go to Neon Console

1. **Visit**: https://console.neon.tech
2. **Select** your `codeblanket` project
3. **Click** "Settings" in the left sidebar
4. **Scroll down** to find **"IP Allow"** or **"Allowed IPs"** section

### Option A: Disable IP Restrictions (Recommended for Development)

1. Find the toggle for **"Enable IP restrictions"** or **"Protect database with IP allowlist"**
2. **Turn it OFF** (disable it)
3. **Save changes**

This allows connections from anywhere, which is fine for development and small apps.

### Option B: Add Specific IPs (More Secure)

If you want to keep IP restrictions enabled:

#### For Development (Your Computer)

1. Find your IP: https://whatismyipaddress.com
2. Add your IP to the allowlist (e.g., `123.456.789.0`)

#### For Production (Vercel)

Add Vercel's IP ranges. You have two options:

**Option 1: Allow All (0.0.0.0/0)**

- This effectively disables IP restrictions
- Simplest solution

**Option 2: Add Vercel Specific IPs**

- Vercel uses dynamic IPs, so you'd need to add many ranges
- Not recommended (too complex)
- Better to use other security methods (auth, row-level security)

## Recommended Settings

For most apps like yours:

```
IP Allow: DISABLED (or allow 0.0.0.0/0)
```

**Why?**

- ✅ Works from development (localhost)
- ✅ Works from production (Vercel)
- ✅ Your app already has authentication (Stack Auth)
- ✅ Row-level security (user_id scoping in queries)
- ✅ SSL/TLS encryption (Neon uses SSL by default)

## After Making Changes

### Test Immediately

1. **Go back to**: https://your-app.vercel.app/debug (or localhost:3000/debug)
2. **Click "Test /api/progress"** again
3. **Should now return**:
   ```json
   {
     "status": 200,
     "data": { "data": {} }
   }
   ```
4. **Click "Test /api/videos"** again
5. **Should now return**:
   ```json
   {
     "status": 200,
     "data": { "videos": [] }
   }
   ```

### Then Test Saving

1. **Click "Test Save to PostgreSQL"**
2. **Should return**: `{ "success": true }`
3. **Go to Neon Console** → SQL Editor
4. **Run**: `SELECT * FROM user_progress;`
5. **See your data!** 🎉

## Alternative: Connection Pooling (If IP Restrictions Required)

If your organization REQUIRES IP restrictions:

### Use Neon Connection Pooler

1. In Neon Console, copy the **POOLER** connection string (not the direct one)
2. It looks like: `postgresql://...pooler.us-east-1.aws.neon.tech/...`
3. Update DATABASE_URL in Vercel to use the pooler URL

The pooler has a static IP that you can allowlist.

## Security Note

**You're already secure without IP restrictions because:**

1. **Stack Auth**: Only authenticated users access APIs
2. **User scoping**: Queries filter by `user_id` (users can only see their own data)
3. **SSL/TLS**: All connections encrypted
4. **Environment variables**: Credentials not exposed to client
5. **API routes**: Server-side only, not accessible without auth

IP restrictions are **extra** security, not required for most apps.

## Quick Checklist

After disabling IP restrictions:

- [ ] Test /api/progress → Returns 200 ✓
- [ ] Test /api/videos → Returns 200 ✓
- [ ] Test Save → Returns success ✓
- [ ] Complete a problem → Saves to PostgreSQL ✓
- [ ] Record a video → Saves to Vercel Blob + metadata to Neon ✓
- [ ] Sign out/in → Data persists ✓

## Still Blocked?

If you still see the error after disabling IP restrictions:

1. **Wait 30 seconds** - Neon settings take a moment to propagate
2. **Hard refresh** your browser (Cmd+Shift+R or Ctrl+Shift+R)
3. **Check Neon status page**: https://status.neon.tech
4. **Try the pooler connection string** instead of direct connection

## Summary

**The fix**: Disable IP restrictions in Neon Console → Settings → IP Allow

This is the #1 cause of "blocked network" errors and is perfectly safe for your use case! 🔓
