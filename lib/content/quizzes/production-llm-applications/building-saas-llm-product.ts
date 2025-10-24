export const buildingSaasLlmProductQuiz = [
  {
    id: 'pllm-q-14-1',
    question:
      'Design the complete architecture for a multi-tenant LLM SaaS product with 3 pricing tiers. Include subscription management, usage tracking, billing, admin dashboard, and tenant isolation. How do you ensure scalability and data security?',
    sampleAnswer:
      'Architecture: API Gateway → Tenant Resolution (from subdomain/API key) → Multi-tenant App Servers → Shared Database with Row-Level Security → Shared Redis Cache → Shared Vector DB with namespaces. Pricing tiers: Free (1K requests/month, GPT-3.5 only), Pro ($49/month, 50K requests, GPT-4 access), Enterprise ($499/month, unlimited, priority support, custom models). Subscription management: Integrate Stripe for payments, create Customer on signup, create Subscription with price_id, handle webhooks (payment_succeeded, subscription_cancelled), store subscription_status in database. Usage tracking: Record every request with tenant_id, aggregate daily in UsageRecords table, check limits before processing (reject if exceeded), display usage dashboard per tenant. Billing: Stripe automatically charges monthly, send usage invoices via email, provide billing history, allow plan upgrades/downgrades (prorate automatically). Admin dashboard: Total revenue, active subscriptions by tier, churn rate, average revenue per user, usage statistics, top tenants by usage, system health metrics. Tenant isolation: Separate database schemas per tenant (better isolation) OR row-level security with tenant_id filter (easier management), use RLS policies: CREATE POLICY tenant_isolation ON conversations FOR ALL TO app_role USING (tenant_id = current_setting(app.current_tenant_id)), encrypt data at rest with tenant-specific keys. Scalability: Horizontal scaling of app servers, database read replicas, Redis cluster, partition large tenants to separate infrastructure.',
    keyPoints: [
      'Multi-tenant architecture with shared infrastructure and RLS for isolation',
      'Stripe integration for subscription and billing management',
      'Comprehensive usage tracking with tier-based limits and admin visibility',
    ],
  },
  {
    id: 'pllm-q-14-2',
    question:
      'Design an onboarding flow for new SaaS customers that maximizes conversion while collecting necessary information. Include email verification, payment setup, API key generation, and initial product guidance.',
    sampleAnswer:
      'Onboarding flow: 1) Landing page → Sign up form (email, company name, password), 2) Email verification (send code, verify within 15min), 3) Choose plan (show pricing, highlight pro), 4) Payment setup if paid plan (Stripe Elements for card, 3D Secure), 5) Create tenant (generate subdomain, API key, setup database), 6) Welcome page with quickstart guide (API documentation, example requests, SDK downloads), 7) Invite team members (optional), 8) First API call tutorial (interactive playground), 9) Setup webhook for notifications (optional). Email verification: Send 6-digit code, 15min expiry, allow resend after 1min, max 3 attempts. Payment: Use Stripe Elements, handle errors gracefully (insufficient funds, declined card), offer trial without card for Free plan, require card for Pro (prevent fraud). Tenant creation: Generate unique subdomain (company-name-123), create API key with sk_ prefix, provision database schema, seed with example data, send welcome email with credentials. Maximize conversion: Offer 14-day free trial on Pro, no credit card required initially, provide interactive demo, highlight ROI (You ll save 40 hours/month), testimonials, live chat support during signup. Reduce friction: Social login (Google, GitHub), skip optional steps, progress indicator, save partially completed signups, email reminders for incomplete signups. Track conversion: Signup → Email verified → Plan selected → Payment → First API call, identify and optimize bottlenecks.',
    keyPoints: [
      'Streamlined flow with progressive disclosure and optional steps',
      'Stripe payment integration with error handling and trial period',
      'Conversion optimization through friction reduction and social proof',
    ],
  },
  {
    id: 'pllm-q-14-3',
    question:
      'How would you build an admin dashboard for managing a multi-tenant LLM SaaS product? What metrics, controls, and features would you include? How do you balance power with simplicity?',
    sampleAnswer:
      'Admin dashboard sections: 1) Overview: Total revenue (MRR, ARR), active tenants, new signups this month, churn rate, average revenue per user, LTV:CAC ratio, system health (uptime, error rate). 2) Tenants: List all tenants with filters (tier, usage, status), search by name/email, view individual tenant details (subscription, usage, API calls, costs, team members), manually adjust limits, add credits, force password reset, impersonate tenant (for support). 3) Usage & Costs: Total API calls per day/month, cost breakdown by model, usage by tier, top 10 highest usage tenants, projected monthly costs, cost per tenant analytics, identify high-cost outliers. 4) Support: Recent support tickets, flagged tenants (payment issues, high error rates), quick actions (send email, adjust subscription, grant temporary access). 5) System: Service status (API, database, cache, queue), error rates, response times, queue depths, cache hit rates, deploy status, scheduled maintenance. Features: Bulk operations (email all Pro users, adjust limits for tier), export data (CSV/Excel reports), audit log (who did what when), alert configuration (email on specific events), custom reports (usage patterns, revenue forecasts). Controls: Tenant management (suspend, delete, modify limits), subscription management (upgrade, downgrade, refund), feature flags (enable beta features per tenant), API key management (view, rotate, revoke). Balance complexity: Default view shows key metrics only, drill-down for details, keyboard shortcuts for power users, save custom views, role-based access (support sees only tenant info, finance sees only revenue, engineers see only technical metrics).',
    keyPoints: [
      'Comprehensive sections: overview, tenants, usage, support, system',
      'Power user features: bulk operations, audit logs, custom reports',
      'Balanced UI with default simplicity and drill-down for details',
    ],
  },
];
