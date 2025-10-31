/**
 * Loading state for auth handler routes
 */

export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#282a36]">
      <div className="text-center">
        <div className="mx-auto h-12 w-12 animate-spin rounded-full border-t-2 border-b-2 border-[#bd93f9]"></div>
        <p className="mt-4 text-[#f8f8f2]">Loading...</p>
      </div>
    </div>
  );
}
