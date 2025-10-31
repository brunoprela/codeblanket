/**
 * Stack Auth configuration for server-side operations
 */

import { StackServerApp } from '@stackframe/stack';

if (!process.env.NEXT_PUBLIC_STACK_PROJECT_ID) {
  throw new Error('NEXT_PUBLIC_STACK_PROJECT_ID is required');
}

if (!process.env.STACK_SECRET_SERVER_KEY) {
  throw new Error('STACK_SECRET_SERVER_KEY is required');
}

export const stackServerApp = new StackServerApp({
  tokenStore: 'nextjs-cookie',
  projectId: process.env.NEXT_PUBLIC_STACK_PROJECT_ID,
  publishableClientKey: process.env.NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY,
  secretServerKey: process.env.STACK_SECRET_SERVER_KEY,
  urls: {
    signIn: '/handler/sign-in',
    afterSignIn: '/',
    afterSignOut: '/',
  },
});
