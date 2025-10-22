/**
 * Quiz questions for Prompt Templates & Variables section
 */

export const prompttemplatesQuiz = [
    {
        id: 'q1',
        question:
            'Explain why hard-coding prompts throughout your codebase is problematic for production applications. Describe the specific issues this creates and how a template system solves them.',
        sampleAnswer:
            'Hard-coding prompts creates severe maintainability and quality problems: (1) Update difficulty - improving a prompt requires finding and updating every instance across the codebase. Missing even one instance creates inconsistent behavior. In large applications, prompts might appear in 50+ places. (2) No version control - cannot track prompt changes over time or rollback when new prompts perform worse. Cannot A/B test prompt variations easily. (3) Duplication and divergence - similar prompts get copy-pasted, then modified slightly, creating maintenance nightmares. "User summarization" prompt exists in 5 variations that should be identical. (4) Testing difficulty - cannot test prompt changes in isolation. Must find every usage and test them all. (5) No reusability - same prompt logic recreated multiple times with slight differences. Cannot leverage improvements across all uses. Template systems solve these by: (1) Centralization - one source of truth for each prompt type. Update once, affects all uses. (2) Versioning - track prompt changes in version control, easy to rollback or compare versions. (3) Variables and composition - create flexible prompts that adapt to different inputs without duplication. Template "summarize" works for articles, emails, documents with just variable changes. (4) A/B testing - swap template versions easily, measure quality differences, promote winners. (5) Documentation - templates can include metadata explaining purpose, required variables, expected output. Real example: E-commerce company had "product description" prompt hard-coded in 15 places. When they improved the prompt, they only updated 12 places, missing 3. Caused inconsistent product descriptions until caught in production. With templates: one change, all 15 uses updated automatically. Template systems are not optional for production - they are as essential as functions for code reuse.',
        keyPoints: [
            'Hard-coded prompts impossible to maintain at scale',
            'Template systems enable centralization and versioning',
            'Variables allow reuse across contexts',
            'A/B testing requires template system',
            'Essential for production quality and maintainability'
        ]
    },
    {
        id: 'q2',
        question:
            'Compare simple Python f-strings versus Jinja2 templates for prompt management. When would you choose each approach, and what are the trade-offs?',
        sampleAnswer:
            'F-strings (simple approach): Best for straightforward variable substitution with no logic. Advantages: (1) Native Python, no dependencies, (2) Simple and fast, (3) Easy to understand, (4) Type safety with Python, (5) Good for short, simple templates. Example: f"Summarize this {doc_type}: {content}". Limitations: (1) No conditional logic - cannot do "if premium user, add extra instructions", (2) No loops - cannot iterate over lists of examples, (3) No template inheritance - cannot extend base templates, (4) No filters - cannot transform variables easily, (5) Harder to load from files. Jinja2 (powerful approach): Best for complex templates with logic, composition, inheritance. Advantages: (1) Conditional logic - {% if premium %}Show advanced analysis{% endif %}, (2) Loops - {% for example in examples %}{{ example }}{% endfor %}, (3) Template inheritance - base templates with child overrides, (4) Filters - {{ text|truncate(100) }}, (5) Load from files easily, (6) Industry standard (used by Flask, Ansible). Limitations: (1) External dependency, (2) Slightly slower than f-strings, (3) Learning curve, (4) Less type safety. When to choose: Use f-strings for: (1) Simple prompts with just variable substitution, (2) Performance-critical paths (though difference is tiny), (3) Teams uncomfortable with template engines. Use Jinja2 for: (1) Complex prompts with conditional sections, (2) Prompts with repeated elements (examples, instructions), (3) Template reuse and inheritance, (4) Loading prompts from external files, (5) Large applications with many prompts. Hybrid approach (recommended): Use f-strings for simple cases (70% of prompts), use Jinja2 for complex cases (30% of prompts), wrap both in unified template manager for consistency. This gets simplicity where possible, power where needed.',
        keyPoints: [
            'F-strings good for simple variable substitution',
            'Jinja2 needed for conditional logic and loops',
            'F-strings faster and simpler, Jinja2 more powerful',
            'Choose based on template complexity',
            'Hybrid approach gets best of both'
        ]
    },
    {
        id: 'q3',
        question:
            'Design a template system for a production application that needs to support template versioning, A/B testing, and gradual rollout of new prompts. What features and architecture would you implement?',
        sampleAnswer:
            'Production template system architecture: Core components: (1) Template registry - central storage for all templates with versions, (2) Version manager - handles multiple versions per template, (3) A/B testing engine - routes users to template variants, (4) Metrics collector - tracks quality by template version, (5) Rollout controller - gradually shifts traffic to new versions. Data model: Template: {name, versions[], default_version, metadata}. Version: {version_id, template_string, required_vars, created_date, quality_score, traffic_percentage}. Features to implement: (1) Template versioning - every change creates new version, never overwrite existing versions, keep version history for rollback, tag versions (v1.0, v1.1-beta, v2.0). (2) A/B testing - assign users to variants (hash user_id for consistency), track metrics per variant (quality, cost, latency), statistical significance testing, automatic winner promotion. (3) Gradual rollout - start new version at 1% traffic, monitor quality metrics, automatically increase to 5%, 10%, 25%, 50%, 100% if metrics good, rollback to previous version if metrics degrade. (4) Feature flags - enable/disable templates per environment (dev/staging/prod), per user segment (free/paid), per region. (5) Validation - validate required variables before render, type checking for variables, schema validation for template structure. Implementation: Store templates in database with version control, cache active templates in Redis for performance, expose API for template management, dashboard for monitoring quality by version, alerts when template performs poorly. Example workflow: Deploy template v2.0 at 5% traffic, monitor for 24 hours, if quality >= v1.0 and no errors, increase to 25%, continue until 100%, set v2.0 as default. If quality drops, rollback to v1.0 instantly. This system enables safe, data-driven prompt iteration - critical for production where prompt changes can significantly impact user experience and costs.',
        keyPoints: [
            'Version every template change, never overwrite',
            'A/B test with statistical significance',
            'Gradual rollout with automatic promotion/rollback',
            'Track metrics per template version',
            'Database storage with Redis caching for performance'
        ]
    }
];

