/**
 * Quiz questions for Code Comment & Documentation Generation section
 */

export const codecommentdocumentationgenerationQuiz = [
  {
    id: 'bcgs-codedoc-q-1',
    question:
      'Explain the difference between documenting "what" code does vs. "why" it does it. Give examples of each, and describe when generated documentation tends to fail at capturing the "why".',
    hint: 'Consider intent, business logic, historical context, and non-obvious decisions.',
    sampleAnswer:
      '**"What" vs "Why" Documentation:** **"What" Documentation (Easy to Generate):** Describes observable behavior - what the code literally does: ```python\n# WHAT (can be generated)\ndef calculate_discount(price: float, customer_tier: str) -> float:\n    """Calculate discount amount based on customer tier.\n    \n    Args:\n        price: Original price\n        customer_tier: Customer tier (gold/silver/bronze)\n    \n    Returns:\n        Discount amount\n    """``` This explains parameters, return value, basic function. LLMs excel at this. **"Why" Documentation (Hard to Generate):** Explains intent, business decisions, non-obvious choices: ```python\n# WHY (hard to generate without context)\ndef calculate_discount(price: float, customer_tier: str) -> float:\n    """Calculate discount amount based on customer tier.\n    \n    NOTE: We cap discounts at 30% due to historical fraud issues\n    where users exploited higher discounts. See ticket #1234.\n    \n    The tier thresholds were set by marketing in Q3 2023\n    to match competitor pricing strategies.\n    """``` **When Generation Fails at "Why":** 1) **Business Context:** Why this specific algorithm? (Chosen to match accounting system), 2) **Historical Decisions:** Why this workaround? (Bug in dependency, fixed in v2.0, keeping for backward compat), 3) **Non-Obvious Constraints:** Why this limit? (API rate limit, database performance, legal requirement), 4) **Future Intent:** Why structured this way? (Preparing for feature X next quarter). **Example Failure:** ```python\nif timeout > 30:\n    timeout = 30  # Why 30? Legal requirement? Performance? Random?\n``` Generated doc: "Sets timeout to 30 if greater than 30" (What). Needed: "Cap at 30 seconds to comply with data protection regulations requiring timely responses" (Why).',
    keyPoints: [
      '"What" = observable behavior, easy to generate from code',
      '"Why" = intent, business context, decisions - needs human input',
      'Generated docs miss: business reasons, historical context, constraints',
      'Best approach: Generate "what", humans add "why"',
    ],
  },
  {
    id: 'bcgs-codedoc-q-2',
    question:
      'Design a system for keeping generated documentation in sync with code changes. When code is modified, how do you determine if documentation needs updating, and what should be regenerated vs. preserved?',
    hint: 'Consider signature changes, behavior changes, and human-written content.',
    sampleAnswer:
      '**Documentation Sync System:** **1) Detect Changes Requiring Doc Updates** - **Signature Changes:** Function parameters added/removed/renamed → Update Args section, Return type changed → Update Returns section, Function renamed → Update examples using it. **Behavior Changes:** Code logic substantially changed → Regenerate description, New edge cases added → Update examples, Error handling changed → Update Raises section. **2) Classify Documentation Content** - **Auto-Generated (Always Regenerate):** Parameter descriptions (can infer from types/names), Return value description (can infer from type), Basic "what it does" summary. **Human-Written (Always Preserve):** "Why" explanations and business context, Historical notes, Design decisions, Links to tickets/docs, Warnings and gotchas. **3) Smart Update Algorithm:** ```python\ndef update_docs(old_code, new_code, old_docs):\n    changes = detect_changes(old_code, new_code)\n    \n    # Extract sections\n    auto_sections = extract_auto_generated(old_docs)\n    human_sections = extract_human_content(old_docs)\n    \n    # Regenerate auto sections only\n    if changes.signature_changed:\n        auto_sections["args"] = generate_args_doc(new_code)\n        auto_sections["returns"] = generate_returns_doc(new_code)\n    \n    if changes.behavior_changed:\n        auto_sections["description"] = generate_description(new_code)\n    \n    # Merge: human content + updated auto content\n    return merge_documentation(auto_sections, human_sections)``` **4) Markers for Human Content:** Use special markers: ```python\n"""Function description (auto-generated).\n\n# HUMAN: Business Context\nThis uses algorithm X because of requirement Y.\nSee design doc: https://...\n# END HUMAN\n\nArgs:\n    param1: Description (auto-generated)\n"""``` Preserve content between HUMAN markers. **5) User Review:** Flag docs as "needs review" after code changes, Show diff of doc changes, Allow user to approve/modify.',
    keyPoints: [
      'Detect signature vs behavior changes',
      'Regenerate auto-generated parts (params, returns, basic description)',
      'Preserve human-written context (why, business logic, warnings)',
      'Use markers to identify human vs auto-generated content',
    ],
  },
  {
    id: 'bcgs-codedoc-q-3',
    question:
      "You're generating README files for projects. What information should be extracted from code vs. what requires human input? Design a template that balances automation with customization.",
    hint: 'Consider project structure, dependencies, usage examples, and project-specific context.',
    sampleAnswer:
      '**README Generation Strategy:** **Auto-Extract from Code:** 1) **Project Structure** - Parse directory tree, identify main modules, detect patterns (MVC, microservices). 2) **Dependencies** - Parse requirements.txt, package.json, go.mod, list versions. 3) **Entry Points** - Find main.py, server.py, CLI commands (from Click/argparse). 4) **API Endpoints** - Parse route decorators (@app.route, @api.get), extract HTTP methods and paths. 5) **Installation Steps** - Infer from project type (Python → pip install, Node → npm install). **Require Human Input:** 1) **Project Purpose** - Why does this exist? What problem does it solve? 2) **Getting Started** - Quick start guide, common workflows. 3) **Configuration** - Environment variables, config files, what they mean. 4) **Architecture Decisions** - Why this structure? Key design choices. 5) **Contributing Guidelines** - How to contribute, code standards. **Template with Mix:** ```markdown\n# {PROJECT_NAME} (auto: from package metadata)\n\n{HUMAN: One-line description}\n\n## Features (auto: from code analysis)\n- REST API with {X} endpoints\n- Database: {detected_db}\n- Authentication: {detected_auth}\n\n{HUMAN: Add key features}\n\n## Installation (auto: from project type)\n\npip install -r requirements.txt  # Auto-detected\n\n{HUMAN: Add setup steps}\n\n## Usage (hybrid)\n\n### Quick Start (HUMAN)\n{User provides example}\n\n### API Endpoints (auto: from route analysis)\n- GET /api/users - List users\n- POST /api/users - Create user\n[...]\n\n## Configuration (HUMAN)\n[User documents env vars]\n\n## Project Structure (auto)\n[Generate from file tree]\n\n## Contributing (HUMAN)\n[User provides guidelines]\n``` **Interactive Generation:** ```python\ndef generate_readme(project_path):\n    auto_data = extract_project_info(project_path)\n    \n    human_prompts = {\n        "description": "Brief project description:",\n        "key_features": "List key features (optional):",\n        "quick_start": "Quick start example:",\n        "configuration": "Configuration docs:"  \n    }\n    \n    human_data = prompt_user(human_prompts)\n    \n    return render_template(auto_data, human_data)``` This balances automation (structure, endpoints, dependencies) with human context (purpose, usage, architecture).',
    keyPoints: [
      'Auto-extract: structure, dependencies, API endpoints, installation',
      'Human input: purpose, getting started, config details, architecture',
      'Use template with placeholders for human customization',
      'Interactive generation prompts for required human content',
    ],
  },
];
