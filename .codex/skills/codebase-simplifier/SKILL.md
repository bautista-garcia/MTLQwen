---
name: codebase-simplifier
description: Simplify an existing codebase by removing redundant code and unnecessary complexity while preserving behavior. Use for refactors intended to reduce real code and conceptual surface area; do not use for feature work or cosmetic file reorganization.
---

# Codebase Simplifier

Make the system genuinely smaller and easier to understand. A lower line count is evidence only when it comes from deleting duplicated logic, obsolete paths, needless abstractions, state, branches, conversions, or intermediates.

## Operating rules

- Establish the behavior and invariants of the affected code before changing it. Inspect callers, tests, and adjacent implementations enough to distinguish a duplicate from an intentional variant.
- Prefer deletion and consolidation. Remove dead code, duplicate implementations, redundant validation or conversion layers, pass-through wrappers, special cases already covered by a general path, and state that can be derived locally.
- When two paths differ only in parameters or a small policy choice, unify them in the existing implementation if the result remains direct and readable. Do not introduce a framework, hierarchy, or generic helper merely to avoid a few repeated lines.
- Keep the control flow linear. Flatten branches and remove temporary values only when the resulting code remains clear and preserves ordering, lifetime, error, and concurrency semantics.
- Treat comments, tests, documentation, build configuration, and call sites as part of the simplification: delete or update material made obsolete by the change.

## Anti-reward-hacking constraint

Do not claim a simplification by moving code around. In particular, do not split a source file into multiple files, move code into wrappers or helpers, rename complexity, or hide it behind generated code solely to lower per-file or diff line counts. Add a file or extraction only when it independently improves a stable boundary, ownership, or reuse; explain that reason and do not count it as the simplification itself.

Run the repository's formatter on both the baseline and the final code before measuring line reduction, and leave changed files formatted. Report only the formatted line-count difference. Joining independent statements, declarations, branches, loops, or functions onto fewer physical lines never counts as simplification. If the repository has no formatter, preserve its established layout and exclude formatting-only changes from the reported reduction.

Do not trade behavior, diagnostics, safety checks required by real inputs, performance-critical invariants, or test coverage for fewer lines without explicit authorization.

## Method

1. Choose a bounded target and write down what behavior must remain unchanged.
2. Locate concrete duplication and complexity using call graphs, tests, and history only as needed. Prefer the smallest change that removes the underlying cause.
3. Implement the deletion or consolidation in place. Keep public contracts stable unless the request authorizes a contract change.
4. Verify the closest relevant tests, build or type checks, and a focused review of changed call paths. Compare before and after conceptually: fewer independent paths, states, or rules should remain.
5. Report what was removed, what was unified, the verification run, and any intentionally retained complexity with its reason. State explicitly if no honest simplification was found rather than manufacturing one.
