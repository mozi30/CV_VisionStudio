---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Governance**: Implementation tasks must include applicable security,
dependency, error-handling, AI-review, and release-gate work required by the
supplemental constitution.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /speckit.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/
  - Security, dependency, error-handling, AI-review, and release-gate requirements
    required by the supplemental constitution

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure required tests, linting, formatting, and release-gate commands

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T004 Setup database schema and migrations framework
- [ ] T005 [P] Implement authentication/authorization framework
- [ ] T006 [P] Setup API routing and middleware structure
- [ ] T007 Create base models/entities that all stories depend on
- [ ] T008 Configure error handling and logging infrastructure
- [ ] T009 Setup environment configuration management
- [ ] T010 Document trust boundaries, secret handling, and sensitive logging constraints
- [ ] T011 [P] Review dependency additions, licenses, vulnerabilities, version policy, and removal strategies
- [ ] T012 [P] Define expected error categories, retries, timeouts, cleanup behavior, and diagnostics

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 1 ⚠️

> **NOTE: Include tests for functional behavior plus applicable security,
> dependency integration, and error paths.**

- [ ] T013 [P] [US1] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T014 [P] [US1] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T015 [P] [US1] Error-path test for [failure mode] in tests/unit/test_[name].py
- [ ] T016 [P] [US1] Security or input-validation test for [trust boundary] in tests/unit/test_[name].py

### Implementation for User Story 1

- [ ] T017 [P] [US1] Create [Entity1] model in src/models/[entity1].py
- [ ] T018 [P] [US1] Create [Entity2] model in src/models/[entity2].py
- [ ] T019 [US1] Implement [Service] in src/services/[service].py with explicit error handling
- [ ] T020 [US1] Implement [endpoint/feature] in src/[location]/[file].py with input validation and safe logging
- [ ] T021 [US1] Document dependency, AI-review, and release-gate evidence for user story 1

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 2 ⚠️

- [ ] T022 [P] [US2] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T023 [P] [US2] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T024 [P] [US2] Error-path or security test for [scenario] in tests/unit/test_[name].py

### Implementation for User Story 2

- [ ] T025 [P] [US2] Create [Entity] model in src/models/[entity].py
- [ ] T026 [US2] Implement [Service] in src/services/[service].py with explicit error handling
- [ ] T027 [US2] Implement [endpoint/feature] in src/[location]/[file].py with input validation and safe logging
- [ ] T028 [US2] Integrate with User Story 1 components (if needed)
- [ ] T029 [US2] Document dependency, AI-review, and release-gate evidence for user story 2

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 3 ⚠️

- [ ] T030 [P] [US3] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T031 [P] [US3] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T032 [P] [US3] Error-path or security test for [scenario] in tests/unit/test_[name].py

### Implementation for User Story 3

- [ ] T033 [P] [US3] Create [Entity] model in src/models/[entity].py
- [ ] T034 [US3] Implement [Service] in src/services/[service].py with explicit error handling
- [ ] T035 [US3] Implement [endpoint/feature] in src/[location]/[file].py with input validation and safe logging
- [ ] T036 [US3] Document dependency, AI-review, and release-gate evidence for user story 3

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Security review and risk waiver documentation if needed
- [ ] TXXX Dependency vulnerability and license review
- [ ] TXXX Error-path and cleanup verification
- [ ] TXXX AI-generated output review and external fact verification
- [ ] TXXX Release notes, migration notes, and rollback or recovery guidance
- [ ] TXXX Performance optimization across all stories
- [ ] TXXX [P] Additional tests in tests/ for security, error, or release-critical behavior
- [ ] TXXX Run required tests, linting, formatting, type checks where applicable, security checks, and dependency checks
- [ ] TXXX Run quickstart.md validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Required tests before implementation
- Models before services
- Services before endpoints
- Error contracts before error-handling implementation
- Security and dependency reviews before release acceptance
- AI-generated output review before acceptance
- Release gates before human-owner approval
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for [endpoint] in tests/contract/test_[name].py"
Task: "Integration test for [user journey] in tests/integration/test_[name].py"

# Launch all models for User Story 1 together:
Task: "Create [Entity1] model in src/models/[entity1].py"
Task: "Create [Entity2] model in src/models/[entity2].py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Run formatter before review
- Run linter before acceptance
- Triage dependency vulnerabilities before release
- Document security, dependency, error-handling, AI, and release exceptions
- Confirm human-owner approval before release
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
