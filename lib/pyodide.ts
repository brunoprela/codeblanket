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

      if (!window.loadPyodide) {
        throw new Error(
          'Pyodide script not loaded. Make sure to include it in your HTML.',
        );
      }

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
