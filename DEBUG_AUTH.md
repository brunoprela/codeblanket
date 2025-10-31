# Debug Authentication & API

Run these commands in your browser console to diagnose the issue:

## Test 1: Check Authentication

```javascript
fetch('/api/auth/check')
  .then(r => r.json())
  .then(data => {
    console.log('Auth check result:', data);
    if (data.authenticated) {
      console.log('✅ Authenticated as:', data.userId);
    } else {
      console.log('❌ Not authenticated');
    }
  })
  .catch(err => console.error('Auth check failed:', err));
```

**Expected**: `{ authenticated: true, userId: "..." }`

## Test 2: Test Progress API

```javascript
fetch('/api/progress')
  .then(async r => {
    console.log('Status:', r.status);
    const data = await r.json();
    console.log('Response:', data);
    return data;
  })
  .catch(err => console.error('Progress API failed:', err));
```

**Expected**: `{ data: { ... } }` or empty `{ data: {} }`

## Test 3: Test Videos API

```javascript
fetch('/api/videos')
  .then(async r => {
    console.log('Status:', r.status);
    const data = await r.json();
    console.log('Response:', data);
    return data;
  })
  .catch(err => console.error('Videos API failed:', err));
```

**Expected**: `{ videos: [] }` (empty initially)

## Possible Issues & Solutions

### Issue 1: Status 401 (Unauthorized)
**Means**: Stack Auth not recognizing your session
**Fix**: 
- Sign out and sign in again
- Clear cookies and retry
- Check Stack Auth dashboard for OAuth configuration

### Issue 2: Status 500 (Server Error)
**Means**: Database query failed
**Check**:
```javascript
// The response should have details
fetch('/api/progress')
  .then(r => r.json())
  .then(data => console.log('Error details:', data.details));
```

Common causes:
- Database connection issue
- Missing environment variable
- SQL query error

### Issue 3: Network Error
**Means**: Can't reach API
**Check**:
- Dev server is running
- No CORS issues
- Correct port (3000)

## Quick Fixes

### If "Unauthorized" (401):

1. **Check if you're actually logged in**:
```javascript
// In browser console
document.cookie
// Should see Stack Auth cookies
```

2. **Try signing out and back in**:
   - Click user menu → Sign Out
   - Click Sign In
   - Sign in with Google again

### If "Server Error" (500):

1. **Check server logs** in your terminal where `npm run dev` is running

2. **Test database connection**:
   - Go to Neon Console
   - Run: `SELECT 1;`
   - Should return `1`

3. **Check environment variables**:
```bash
# In terminal
cat frontend/.env
```

Should show all variables without quotes

## Working State Checklist

- [ ] `fetch('/api/auth/check')` returns `authenticated: true`
- [ ] `fetch('/api/progress')` returns 200 status
- [ ] `fetch('/api/videos')` returns 200 status
- [ ] Database tables exist (user_progress, user_videos)
- [ ] Environment variables set correctly (no quotes)
- [ ] Dev server running without errors

## Still Not Working?

Run all tests above and share the results. The error messages will tell us exactly what's wrong!

