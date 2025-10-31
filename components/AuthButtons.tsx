'use client';

/**
 * Authentication buttons for Stack Auth
 * Shows sign in button when logged out, user info when logged in
 */

import { useUser } from '@stackframe/stack';
import { useState } from 'react';
import Link from 'next/link';

export default function AuthButtons() {
  const user = useUser();
  const [showDropdown, setShowDropdown] = useState(false);

  if (!user) {
    return (
      <Link
        href="/handler/sign-in"
        className="rounded-md bg-[#bd93f9] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#a070e0]"
      >
        Sign In
      </Link>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="flex items-center gap-2 rounded-md bg-[#44475a] px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-[#6272a4]"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#bd93f9] text-sm font-bold">
          {user.displayName?.[0]?.toUpperCase() ||
            user.primaryEmail?.[0]?.toUpperCase() ||
            'U'}
        </div>
        <span className="hidden sm:inline">
          {user.displayName || user.primaryEmail || 'User'}
        </span>
      </button>

      {showDropdown && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setShowDropdown(false)}
          />
          <div className="absolute right-0 z-20 mt-2 w-48 rounded-md border border-gray-600 bg-[#282a36] py-1 shadow-lg">
            <div className="border-b border-gray-600 px-4 py-2">
              <p className="text-sm font-medium text-white">
                {user.displayName || 'User'}
              </p>
              <p className="text-xs text-gray-400">{user.primaryEmail}</p>
            </div>
            <Link
              href="/handler/account-settings"
              className="block px-4 py-2 text-sm text-white transition-colors hover:bg-[#44475a]"
            >
              Account Settings
            </Link>
            <Link
              href="/handler/sign-out"
              className="block px-4 py-2 text-sm text-red-400 transition-colors hover:bg-[#44475a]"
            >
              Sign Out
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
