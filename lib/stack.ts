/**
 * Stack Auth configuration for server-side operations
 */

import { StackServerApp } from '@stackframe/stack';

// Use placeholder values during build if env vars not available
// This allows the build to succeed, runtime will use actual values
const projectId =
  process.env.NEXT_PUBLIC_STACK_PROJECT_ID || 'build-placeholder';
const publishableClientKey =
  process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY || 'build-placeholder';
const secretServerKey =
  process.env.STACK_SECRET_SERVER_KEY || 'build-placeholder';

// Warn if using placeholders (only happens during build without env vars)
if (projectId === 'build-placeholder' && typeof window === 'undefined') {
  console.warn(
    'Stack Auth using placeholder credentials during build. Set environment variables for production.',
  );
}

export const stackServerApp = new StackServerApp({
  tokenStore: 'nextjs-cookie',
  projectId,
  publishableClientKey,
  secretServerKey,
  urls: {
    signIn: '/handler/sign-in',
    afterSignIn: '/',
    afterSignOut: '/',
  },
});
