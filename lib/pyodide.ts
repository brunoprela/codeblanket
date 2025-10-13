// Pyodide loader - loads Python runtime once and reuses it
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pyodideInstance: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadingPromise: Promise<any> | null = null;

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    loadPyodide: any;
  }
}

/**
 * Wait for window.loadPyodide to be available
 * @param timeout Maximum time to wait in milliseconds
 * @returns Promise that resolves when loadPyodide is available
 */
async function waitForPyodideScript(timeout = 10000): Promise<void> {
  const startTime = Date.now();

  while (!window.loadPyodide) {
    if (Date.now() - startTime > timeout) {
      throw new Error(
        'Pyodide script not loaded. Make sure you have an internet connection and try refreshing the page.',
      );
    }
    // Wait 100ms before checking again
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

export async function getPyodide() {
  // If already loaded, return it
  if (pyodideInstance) {
    return pyodideInstance;
  }

  // If currently loading, wait for that promise
  if (loadingPromise) {
    return loadingPromise;
  }

  // Start loading
  loadingPromise = (async () => {
    try {
      if (typeof window === 'undefined') {
        throw new Error('Pyodide can only be loaded in the browser');
      }

      // Wait for the Pyodide script to be available
      await waitForPyodideScript();

      pyodideInstance = await window.loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
      });

      return pyodideInstance;
    } catch (error) {
      loadingPromise = null;
      throw error;
    }
  })();

  return loadingPromise;
}
