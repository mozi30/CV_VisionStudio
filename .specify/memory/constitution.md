<!--
Sync Impact Report
Version change: 2.0.0 → 3.0.0
Modified principles:
- I. Readability Supremacy → I. Security Constitution
- II. Consistency First → II. Dependency Management Constitution
- III. Explicit Layout → III. Error Handling Constitution
- IV. Import Discipline → IV. AI Usage Constitution
- V. Naming Is Contract → V. Release Gates Constitution
Added sections:
- Purpose
- VI. Unified Acceptance Rule
Removed sections:
- Source Authority
- VI. Whitespace Has Meaning
- VII. Comments Explain Why
- VIII. Pythonic Expressions
- IX. Public API Stability
- X. Tool-Enforced Style
- Required Workflow
- Acceptance Criteria
- Non-Negotiables
Templates requiring updates:
- ✅ .specify/templates/plan-template.md
- ✅ .specify/templates/spec-template.md
- ✅ .specify/templates/tasks-template.md
- ✅ .specify/templates/checklist-template.md
- ✅ .specify/templates/commands/ (directory absent; no command templates to update)
Follow-up TODOs:
- None
-->
# Vision Studio Constitution

## Purpose

This constitution defines the project rules that support the core Spec Kit
workflow. The core Spec Kit workflow governs specification, testing,
verification, adversarial review, and traceability. This supplemental
constitution governs the surrounding engineering discipline required to keep the
project secure, maintainable, reviewable, and releasable over time.

## Core Principles

### I. Security Constitution

Security is a design requirement, not a release-phase checklist.
Security-sensitive behavior MUST be specified, tested, reviewed, and verified
with the same discipline as functional correctness.

Rules:

- Secrets MUST NOT be committed to source control.
- Credentials, tokens, private keys, and API keys MUST be loaded through approved
  secret-management mechanisms.
- Inputs MUST be validated at trust boundaries.
- Authentication MUST be explicit.
- Authorization MUST be checked server-side or at the trusted enforcement
  boundary.
- User-controlled data MUST NOT be trusted.
- Security-sensitive code MUST receive adversarial review.
- Dependency vulnerabilities MUST be triaged before release.
- Cryptography MUST use established, reviewed libraries.
- Custom cryptographic algorithms MUST NOT be implemented.
- Error messages MUST NOT leak secrets, credentials, stack traces, or sensitive
  internals to users.
- Logs MUST NOT contain secrets or private user data unless explicitly approved
  and protected.
- Security exceptions MUST be documented and accepted by the human owner.

Security-sensitive work is acceptable only when threat-relevant behavior is
included in the specification, trust boundaries are identified, input validation
is tested, authentication and authorization behavior is tested, known
vulnerabilities are reviewed, sensitive data handling is intentional, and
security risks are resolved or explicitly waived.

Non-negotiables: no committed secrets, no custom cryptography, no silent
authorization assumptions, and no release with untriaged critical security risk.

### II. Dependency Management Constitution

Every dependency is a liability until justified. Dependencies add maintenance
cost, security risk, licensing exposure, and long-term coupling. They MUST be
introduced deliberately.

Rules:

- New dependencies MUST be justified.
- The standard library SHOULD be preferred when sufficient because it reduces
  supply-chain and maintenance risk.
- Small local implementations SHOULD be preferred over large dependencies for
  trivial behavior because dependency cost must match value.
- Dependencies MUST be actively maintained.
- Dependencies SHOULD have clear documentation and reasonable adoption because
  maintainability and reviewability depend on available context.
- Known vulnerable dependencies MUST be upgraded, replaced, isolated, or
  explicitly waived.
- Lockfiles MUST be committed for applications.
- Version ranges MUST be intentional.
- Unused dependencies MUST be removed.
- Dependencies MUST NOT be added only for convenience if they materially increase
  project risk.
- License compatibility MUST be considered before adding dependencies.
- Transitive dependencies SHOULD be reviewed for high-risk packages because
  high-risk transitive code can affect release safety.

Before adding a dependency, the review MUST answer what problem the dependency
solves, why the standard library is insufficient, whether the package is
maintained, whether the license is acceptable, what security or supply-chain risk
it introduces, and what removal path exists if the dependency becomes unsafe or
abandoned.

A dependency is acceptable only when its purpose is documented, maintenance
status is acceptable, license is compatible, security risk is understood, it is
covered by lockfile or version policy, and it does not duplicate existing project
capability without justification.

Non-negotiables: no unjustified dependencies, no abandoned critical-path
dependency without waiver, no ignored critical vulnerability, and no critical-path
dependency added without a removal strategy.

### III. Error Handling Constitution

Failure is part of the contract. Errors MUST be designed, specified, tested, and
observable. A system that fails unclearly is not complete.

Rules:

- Expected failure modes MUST be specified.
- Errors MUST be explicit, typed where practical, and actionable.
- Exceptions MUST NOT be swallowed silently.
- Generic catch-all handlers MUST be justified.
- User-facing errors MUST be safe and understandable.
- Internal errors MUST include enough diagnostic context for maintainers.
- Retry logic MUST be bounded and intentional.
- Timeouts MUST be explicit for external calls.
- Partial failure behavior MUST be specified.
- Cleanup behavior MUST be tested where resources are acquired.
- Error paths MUST be tested, not merely happy paths.
- Error handling MUST NOT hide corrupted state.
- Logs MUST distinguish expected failures from exceptional failures.

Specifications SHOULD identify relevant error categories because explicit
categories improve coverage and review. Relevant categories include invalid
input, missing input, permission denied, authentication failure, resource not
found, conflict or invalid state, timeout, rate limit, external dependency
failure, internal invariant violation, and data corruption or malformed data.

Error handling is acceptable only when failure modes are included in the
specification, error paths are tested, user-facing messages are safe, internal
diagnostics are useful, retries and timeouts are bounded, no exception is
silently swallowed, and critical cleanup behavior is verified.

Non-negotiables: no silent failure, no unbounded retry loops, no user-facing
secret leakage, and no unspecified critical failure mode.

### IV. AI Usage Constitution

AI accelerates engineering but does not own accountability. AI-generated output
is draft material until reviewed, tested, and accepted by the human owner.

Rules:

- AI-generated code MUST be reviewed before acceptance.
- AI MUST NOT invent requirements.
- AI MUST NOT invent APIs, libraries, file paths, data models, or external facts
  without verification.
- AI MUST preserve project conventions.
- AI MUST follow the accepted specification.
- AI MUST write tests before implementation when operating inside the Spec Kit
  workflow.
- AI MUST disclose uncertainty when requirements are ambiguous.
- AI MUST NOT silently broaden scope.
- AI MUST NOT remove tests or weaken assertions merely to make a build pass.
- AI MUST NOT bypass verification, linting, or review gates.
- AI-generated security-sensitive code MUST receive human and adversarial review.
- Human approval is required for architecture, security, data handling, and
  release decisions.

AI contributors SHOULD be assigned explicit roles because role separation
improves review quality. Roles include Builder for drafting specifications,
tests, implementation, and refactors; Reviewer for checking correctness,
maintainability, and style; Adversary for finding flaws, gaps, and unsafe
assumptions; and Summarizer for recording decisions, risks, and traceability. A
single AI pass SHOULD NOT be treated as sufficient for critical work because
critical work requires independent review pressure.

AI-assisted work is acceptable only when it traces to the specification, has
passing tests, follows project style, has been reviewed, claims about external
facts, APIs, or dependencies are verified, security-sensitive output has received
adversarial review, and the human owner accepts the result.

Non-negotiables: no AI-generated artifact is authoritative by default, no
invented requirements, no weakening tests to satisfy implementation, and no
release without human accountability.

### V. Release Gates Constitution

A feature is not done when code is written. A feature is done when it has passed
the required quality, safety, documentation, and release gates.

Rules:

- Releases MUST pass the configured test suite.
- Releases MUST pass linting and formatting checks.
- Releases SHOULD pass type checks where the language and project support them
  because type checks catch compatibility and integration defects early.
- Required formal verification gates MUST pass.
- Required security checks MUST pass or be explicitly waived.
- Required dependency checks MUST pass or be explicitly waived.
- Required migration checks MUST pass.
- Release notes MUST describe user-visible changes.
- Breaking changes MUST be documented.
- Configuration changes MUST be documented.
- Operational risks MUST be documented.
- Rollback or recovery guidance MUST exist for risky releases.
- Known defects MUST be documented before release.
- Human owner approval is required for release.

Before release, the pre-release checklist MUST confirm specification acceptance,
passing tests, passing linting, passing formatting, passing type checks where
applicable, completed security checks, completed dependency checks, completed
required verification, updated documentation, written release notes, reviewed
migration path, reviewed rollback or recovery path, accepted known risks, and
human-owner approval.

A release is acceptable only when required checks pass, required reviews are
complete, user-visible behavior is documented, breaking changes are explicit,
operational risks are understood, rollback or recovery is possible where needed,
and the human owner accepts the release.

Non-negotiables: no release with failing required tests, no release with
unreviewed critical security risk, no release with undocumented breaking changes,
and no release without human approval.

### VI. Unified Acceptance Rule

A change governed by this constitution is acceptable only when it satisfies all
applicable gates:

```text
Security reviewed
→ Dependencies justified
→ Errors specified and tested
→ AI output reviewed
→ Release gates passed
→ Human owner accepted
```

## Governance

This constitution supplements the core Spec Kit workflow and supersedes
conflicting informal practices in this repository. Security, dependency, error
handling, AI-assisted work, and release decisions MUST comply with this
constitution unless an explicit human-owner waiver is recorded.

Amendments MUST be proposed as a documented change that includes the rationale,
expected impact on existing artifacts, required template updates, migration needs,
and review evidence. Amendments become effective only after human-owner review and
acceptance.

Versioning follows semantic versioning for governance changes:

- MAJOR: backward-incompatible governance changes, principle removals, or
  redefinitions.
- MINOR: new principles, new mandatory sections, or materially expanded guidance.
- PATCH: clarifications, wording improvements, typo fixes, or non-semantic
  refinements.

Compliance review MUST occur during specification review, planning, task
generation, implementation review, and release acceptance. Reviewers MUST verify
security handling, dependency justification, error contracts, AI accountability,
release readiness, and recorded human-owner acceptance.

**Version**: 3.0.0 | **Ratified**: 2026-05-26 | **Last Amended**: 2026-05-26
