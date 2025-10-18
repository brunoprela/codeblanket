/**
 * Quiz questions for API Governance section
 */

export const apigovernanceQuiz = [
  {
    id: 'governance-d1',
    question:
      'You join a company with 50+ APIs, no standards, inconsistent naming, and no central catalog. Design a governance framework to bring order.',
    sampleAnswer: `API Governance Framework Implementation:

**Phase 1: Assessment (Month 1)**
- Inventory all APIs
- Document current state
- Identify inconsistencies

**Phase 2: Standards (Months 2-3)**
- Define API design standards
- Create OpenAPI linting rules
- Document versioning policy
- Establish deprecation timeline

**Phase 3: Tooling (Months 3-4)**
- Implement Spectral for linting
- Create API catalog
- Set up pre-commit hooks
- Add CI/CD checks

**Phase 4: Migration (Months 4-12)**
- Migrate APIs gradually to standards
- Support old + new patterns
- Provide migration guides
- Sunset non-compliant APIs

**Phase 5: Enforcement (Ongoing)**
- API review board
- Automated checks in CI/CD
- Regular audits
- Documentation requirements`,
    keyPoints: [
      'Start with inventory and assessment of current state',
      'Define clear standards and conventions',
      'Implement automated linting and checks',
      'Gradual migration with support for legacy patterns',
      'Ongoing enforcement through review process',
    ],
  },
  {
    id: 'governance-d2',
    question:
      'Two teams want to build similar APIs (user management). How do you decide: build one shared API or separate APIs?',
    sampleAnswer: `Decision framework for shared vs separate APIs:

**Build Shared API When**:
- ✅ Exact same use case
- ✅ Similar SLAs and requirements
- ✅ Teams willing to collaborate
- ✅ Resources (users) truly shared
- ✅ Low political friction

**Build Separate APIs When**:
- ✅ Different use cases (internal vs external)
- ✅ Different SLAs (99.9% vs 99.99%)
- ✅ Different schemas/data models
- ✅ Teams can't collaborate
- ✅ Different domains (identity vs profile)

**Recommendation Process**:
1. API review board meeting
2. Compare requirements
3. Assess team dynamics
4. Consider maintenance burden
5. Make decision, document rationale

**Example**:
- Identity API (auth, login): Shared (central identity)
- Profile API (preferences): Team-specific (different needs)

Balance: Reduce duplication vs team autonomy.`,
    keyPoints: [
      'Consider if use cases and requirements truly overlap',
      'Assess team collaboration willingness',
      'Evaluate SLA and domain requirements',
      'Balance duplication reduction vs team autonomy',
      'API review board makes final decision',
    ],
  },
  {
    id: 'governance-d3',
    question:
      'Compare design-first vs code-first approach to API development. Which would you use and when?',
    sampleAnswer: `Comparison:

**Design-First** (OpenAPI spec → Code):

Pros:
- ✅ Early API review before coding
- ✅ Frontend can mock while backend builds
- ✅ Documentation guaranteed to match
- ✅ Catches design issues early
- ✅ Enables contract testing

Cons:
- ❌ Upfront time investment
- ❌ Spec maintenance alongside code

**Code-First** (Code → OpenAPI spec):

Pros:
- ✅ Faster initial development
- ✅ Spec always matches code
- ✅ No spec maintenance

Cons:
- ❌ No early API review
- ❌ Design issues found late
- ❌ Frontend blocked on backend

**Recommendation**:

**Design-First for**:
- Public APIs (external developers)
- Large organizations (multiple teams)
- Complex APIs (need review)

**Code-First for**:
- Internal APIs (same team)
- Prototypes (iterating quickly)
- Simple CRUD APIs

**Hybrid Approach**:
- Design major changes first
- Code-first for minor updates
- Auto-generate spec from code
- Review generated specs

Most successful organizations use design-first for external APIs, hybrid for internal.`,
    keyPoints: [
      'Design-first enables early review and parallel development',
      'Code-first is faster for simple/internal APIs',
      'Public APIs should use design-first approach',
      'Internal APIs can use code-first with generated specs',
      'Hybrid approach balances benefits of both',
    ],
  },
];
