/**
 * Stack Auth handler route
 * Handles sign-in, sign-up, account settings, and sign-out
 */

import { StackHandler } from '@stackframe/stack';
import { stackServerApp } from '@/lib/stack';

export default function Handler(props: {
  params: Promise<{ stack: string[] }>;
}) {
  return <StackHandler fullPage app={stackServerApp} routeProps={props} />;
}
