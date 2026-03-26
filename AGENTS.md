# RecStore Repository Guidelines for AI Agents

## Purpose and Audience

This document is the repository-wide operating manual for AI coding agents working in `RecStore`.

- It applies to the entire repository unless a user gives a more specific instruction in the current conversation.
- It is written for agents first, not for human onboarding.
- Prefer direct execution over abstract discussion, but do not skip design and verification discipline.

This repository contains a parameter-server-based recommendation system training stack with C++ server/runtime components, Python/PyTorch client integrations, and TorchRec/DLRM model-zoo examples.

## Language and Output Rules

- Write this repository's `AGENTS.md` and agent-facing operating instructions in English.
- Default user-facing conversation output must be in Chinese.
- Default project documentation must be in Chinese unless the user explicitly asks otherwise.
- Default code comments must be in English.

## Commit and Git Rules

- Default commit messages must be in English.
- Prefer Conventional Commits:
  - `feat(scope): ...`
  - `fix(scope): ...`
  - `docs(scope): ...`
  - `refactor(scope): ...`
  - `test(scope): ...`
  - `ci(scope): ...`
- Do not amend commits unless the user explicitly asks for it.
- Never use destructive git commands such as `git reset --hard` or `git checkout --` unless explicitly requested.
- Assume the worktree may be dirty. Do not revert unrelated user changes.

## Project Overview

RecStore is a hybrid C++ and Python codebase centered on large-scale embedding storage, retrieval, and update workflows for recommendation models.

Current high-level areas:

- `src/ps`, `src/storage`, `src/optimizer`, `src/base`, `src/memory`:
  Core server, storage, optimizer, and runtime infrastructure in C++.
- `src/framework` and `src/python/pytorch/recstore`:
  Python-side clients and framework integration layers.
- `src/python/pytorch/torchrec_kv`:
  TorchRec-oriented embedding access path and integration glue.
- `model_zoo/torchrec_dlrm` and `model_zoo/torchrec_2`:
  Model examples, training scripts, and integration experiments.
- `docs/architecture`, `docs/parameter_server`, `docs/storage`:
  Architecture and subsystem documentation.
- `docs/superpowers/specs` and `docs/superpowers/plans`:
  Design specs and implementation plans created during agent-driven development.

Current implementation background, not a permanent contract:

- Some TorchRec and DLRM paths are sensitive to step ordering, prefetch timing, and sparse update visibility.
- Recent changes introduced stricter step-boundary handling in training loops to prioritize correctness.
- Treat current async and overlap behavior as implementation detail to be validated, not as an architectural guarantee.

## Default Development Workflow

For any feature work, behavior change, architecture change, or non-trivial bug fix, follow this order unless the user explicitly asks to skip part of it:

1. Understand the local context first.
2. Confirm or propose the design before implementation.
3. For multi-step work, write or follow a plan.
4. Implement in small, reviewable increments.
5. Verify with the narrowest useful tests first, then broader checks as needed.

Operational rules:

- Do not jump straight into code changes without understanding the existing implementation and nearby constraints.
- Prefer minimal, local changes over broad refactors.
- Preserve established interfaces and behavior unless the task explicitly changes them.
- If a design doc or plan already exists for the task, execute against it rather than improvising a new architecture.
- Do not claim success before running verification that actually exercises the changed behavior.

## Architecture and Invariants

When modifying this repository, preserve clear boundaries between:

- Storage and server behavior
- Python client protocol and semantics
- Model integration glue
- Training-loop scheduling and optimization logic

Agents should optimize for explicit state transitions and clear ownership.

Prefer:

- Passing context explicitly across steps or stages
- Making synchronization points obvious
- Isolating performance optimizations from correctness-critical paths
- Failing loudly when invariants are violated

Avoid:

- Hidden shared mutable state across batches or steps
- Async naming that does not match real execution semantics
- Mixing correctness changes with speculative performance tuning in the same patch unless tightly coupled
- Large refactors that obscure the behavioral change being made

## Review Priorities

When asked to review code, or when self-reviewing before completion, prioritize findings in this order:

1. Correctness and data consistency
2. Ordering, synchronization, and visibility issues in async or distributed paths
3. Training semantics regressions
4. API compatibility and call-site fallout
5. Missing tests or weak verification
6. Observability and debuggability gaps
7. Performance concerns

Specific repository review focus:

- Sparse update visibility across training steps
- Prefetch and read-after-write ordering
- Implicit state carried between batches
- Tensor device, dtype, and shape mismatches
- Fallback path correctness when optimized paths are unavailable
- Background thread lifecycle, shutdown, and exception propagation
- Consistency between Python wrappers and backend behavior

Performance is important, but correctness wins by default. Do not trade away semantics unless the user explicitly asks for that trade-off and the change is documented.

## Coding Standards

General standards:

- Follow existing repository patterns before introducing new abstractions.
- Keep functions focused and boundaries explicit.
- Prefer code that is easy to inspect over clever but opaque logic.
- Use ASCII by default unless the file already requires non-ASCII content.
- Add comments only where intent or invariants are non-obvious.

Python standards:

- Prefer explicit state over implicit module-level coordination.
- Keep fallback paths readable and behaviorally equivalent where intended.
- When handling async-ish workflows, make submission, wait, and consumption semantics obvious from code and naming.
- Avoid introducing silent best-effort behavior in correctness-sensitive code.

C++ standards:

- Preserve existing style and ownership conventions in surrounding code.
- Avoid broad mechanical churn in performance-sensitive or concurrency-heavy modules.
- Make lifecycle, memory ownership, and synchronization intent explicit.

Documentation standards:

- Project docs default to Chinese.
- Keep design docs and plans task-specific.
- Prefer concise, high-signal explanations over exhaustive prose.

## Testing and Verification

Every behavior change should have verification proportionate to its risk.

Minimum expectations:

- Run the most relevant targeted tests for the changed area.
- If the behavior is not already covered, add tests.
- If tests cannot be run in the current environment, say so explicitly and explain why.

For async, distributed, or training-semantics work, favor tests that verify:

- ordering
- visibility
- fallback behavior
- thread/process shutdown
- error propagation

Useful verification layers in this repository include:

- focused Python unit tests under `src/python/pytorch/recstore/unittest`
- model-zoo integration checks under `model_zoo/torchrec_dlrm`
- compiled test targets in `build/`
- server/client smoke tests against a running `ps_server`

### Default PyTorch Client Verification Flow

When the user asks to validate baseline repository operability, or explicitly asks for a PyTorch client integration check, use this default sequence:

1. Confirm `build/` exists at the repository root.
2. Run `make -j` inside `build/`.
3. Return to the repository root and start:
   `./build/bin/ps_server --config_path ./recstore_config.json`
4. Confirm logs include lines similar to:
   - `bRPC Server shard 0 listening on 127.0.0.1:15123`
   - `bRPC Server shard 1 listening on 127.0.0.1:15124`
5. Run `ctest -R pytorch_client_test -VV` inside `build/`.
6. Stop the manually started `ps_server` after the test completes.

Notes:

- `pytorch_client_test` maps to `src/framework/pytorch/python_client/client_test.py`.
- The test connects to the ports defined in `recstore_config.json`, currently `15123` and `15124`.
- If a usable `ps_server` is already running on those ports, the test may reuse it.
- If the environment blocks local socket binding, start the server and run the client tests in an environment without that restriction.
- If `build/` does not exist or the project is not configured, complete the build setup first.

## Safety Rules for Editing

- Read the relevant code before editing it.
- Never overwrite user changes just to make your patch simpler.
- If you encounter unexpected modifications in the same files, work with them unless they directly block the task.
- If a conflict affects correctness and the right resolution is unclear, stop and ask.
- Keep patches scoped to the task at hand.

For agent behavior:

- Do not present hypothetical fixes as completed work.
- Do not say tests pass unless you ran them.
- Do not hide uncertainty. State assumptions and remaining risks clearly.

## Directory Map

Use this map to orient quickly before searching deeper:

- `src/`
  Main source tree.
- `src/ps/`
  Parameter server runtime and serving logic.
- `src/storage/`
  Storage-layer implementations and related abstractions.
- `src/optimizer/`
  Backend optimizer logic.
- `src/framework/`
  Framework integrations beyond the newer Python-side client paths.
- `src/python/pytorch/recstore/`
  Python client, optimizer wrappers, datasets, and tests.
- `src/python/pytorch/torchrec_kv/`
  TorchRec integration path for embedding reads, prefetch, and updates.
- `model_zoo/torchrec_dlrm/`
  DLRM-related experiments, tests, and training scripts.
- `docs/`
  Project documentation.
- `docs/superpowers/specs/`
  Design specifications.
- `docs/superpowers/plans/`
  Implementation plans.

## Final Execution Principle

Be conservative with semantics, aggressive with clarity, and honest about verification.

In this repository, the most common failure mode is not a syntax mistake. It is changing execution order, visibility, or fallback behavior without realizing it. Work accordingly.
