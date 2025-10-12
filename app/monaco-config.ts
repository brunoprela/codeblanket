// Monaco Editor configuration to suppress loader errors
export function configureMonaco() {
    if (typeof window !== 'undefined') {
        // Suppress Monaco loader errors for optional modules
        const originalError = console.error;
        console.error = (...args) => {
            // Create an error to check the current call stack
            const stack = new Error().stack || '';

            // Check if this error is being called from Monaco code
            if (
                stack.includes('monaco-editor') ||
                stack.includes('vs/loader.js') ||
                stack.includes('jsdelivr.net/npm/monaco-editor') ||
                stack.includes('loader.js')
            ) {
                // Silently ignore - this is a Monaco loader error
                return;
            }

            // Also check all arguments for Monaco-related content
            const shouldSuppress = args.some((arg) => {
                if (!arg) return false;

                // Check string messages
                const message = arg?.toString() || '';
                if (
                    message.includes('Loading "stackframe" failed') ||
                    message.includes('monaco-editor') ||
                    message.includes('vs/loader') ||
                    message.includes('[object Event]')
                ) {
                    return true;
                }

                // Check Error objects and their stacks
                if (arg instanceof Error && arg.stack) {
                    if (
                        arg.stack.includes('monaco-editor') ||
                        arg.stack.includes('vs/loader.js') ||
                        arg.stack.includes('jsdelivr.net/npm/monaco-editor')
                    ) {
                        return true;
                    }
                }

                // Check Event objects
                if (typeof arg === 'object' && arg instanceof Event) {
                    return true;
                }

                return false;
            });

            if (shouldSuppress) {
                // Silently ignore Monaco loader errors
                return;
            }

            originalError.apply(console, args);
        };
    }
}
