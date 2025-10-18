/**
 * Multiple choice questions for Proxies (Forward & Reverse) section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const proxiesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main difference between a Forward Proxy and a Reverse Proxy?',
    options: [
      'Forward proxy is faster than reverse proxy',
      'Forward proxy serves clients (hides client from server), reverse proxy serves servers (hides server from client)',
      'Forward proxy only works with HTTP, reverse proxy works with HTTPS',
      'Forward proxy is for internal networks, reverse proxy is for external networks',
    ],
    correctAnswer: 1,
    explanation:
      "Forward Proxy: Serves clients, sits on client-side, hides client IP from servers. Example: Corporate proxy, VPN. Reverse Proxy: Serves servers, sits on server-side, hides server IP from clients. Example: NGINX, load balancer. Clients know about forward proxy (configured in browser), don't know about reverse proxy (transparent).",
  },
  {
    id: 'mc2',
    question:
      'What is SSL Termination, and why would you use it at a reverse proxy?',
    options: [
      'Blocking SSL connections for security',
      'Decrypting HTTPS at the proxy, forwarding HTTP to backends',
      'Encrypting all traffic end-to-end',
      'Removing SSL certificates from servers',
    ],
    correctAnswer: 1,
    explanation:
      'SSL Termination: Reverse proxy decrypts HTTPS (from clients), forwards HTTP to backends (internal network). Benefits: (1) Offload CPU-intensive SSL from backends (30% CPU reduction). (2) Centralized certificate management (one cert on proxy). (3) Simplified backends (no SSL config). Used by: Netflix, Google, AWS. Security: Internal HTTP acceptable (private network/VPC).',
  },
  {
    id: 'mc3',
    question:
      'A corporate network uses a forward proxy. An employee configures their browser to bypass the proxy. What happens?',
    options: [
      'The employee can access the internet normally',
      'The firewall blocks the direct connection (proxy is mandatory)',
      'The proxy automatically reconfigures the browser',
      'The employee gets faster internet access',
    ],
    correctAnswer: 1,
    explanation:
      "Properly configured corporate network: Firewall blocks all direct internet access. Only proxy allowed (whitelist proxy IP). Employee bypassing proxy: Browser tries direct connection. Firewall blocks (no route to internet). Result: No internet access. This enforces proxy usage (employees can't bypass). Some networks allow direct access (poor security).",
  },
  {
    id: 'mc4',
    question:
      'Which of the following is NOT typically a responsibility of a reverse proxy?',
    options: [
      'Load balancing across backend servers',
      'SSL termination',
      'Content filtering (blocking websites)',
      'Caching static content',
    ],
    correctAnswer: 2,
    explanation:
      "Reverse proxy responsibilities: Load balancing (distribute traffic), SSL termination (offload encryption), caching (reduce backend load), compression, security. Content filtering (blocking websites): Forward proxy responsibility (client-side). Reverse proxy doesn't filter based on destination (it proxies to known backends). Option 3 is forward proxy task.",
  },
  {
    id: 'mc5',
    question:
      'Your application uses NGINX as a reverse proxy. Where should SSL certificates be installed?',
    options: [
      'On each backend server',
      'On the NGINX server (proxy) only',
      'On both NGINX and backend servers',
      'On the client browsers',
    ],
    correctAnswer: 1,
    explanation:
      'With SSL termination at reverse proxy: Install certificate on NGINX only (not backends). NGINX: Handles HTTPS from clients, decrypts, forwards HTTP to backends. Backends: Accept HTTP only (no SSL config needed). Benefits: Centralized certificate management, simplified backends, lower CPU on backends. Backends trust NGINX (internal network).',
  },
];
