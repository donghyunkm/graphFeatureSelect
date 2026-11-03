# AGENTS.md — entry point

This repo's durable guide is **`CLAUDE.md`** — read it first (project overview, tech stack, repo layout,
required `gfs` conda env, commands, development guidelines, end-of-session procedure). This file is a thin
pointer so an agent starting from `AGENTS.md` finds everything.

Start here, in order:
1. **`CLAUDE.md`** — how to work in this repo (env, commands, conventions, guardrails). **Read before substantial work.**
2. **`progress.md`** — concise chronological log: latest session at the top, running/pending Slurm jobs,
   prioritized Next Steps, and a Reference appendix of durable-but-volatile detail (architecture internals,
   data-module variants, `trainmode`, the eval-time HalfHop gotcha).
3. **`docs/conclusion.md`** — the top-level results summary (the two headline claims + the specific
   comparisons behind them); **start here for "what did we find."**
4. **`docs/README.md`** — index of all focused write-ups in `docs/` (one topic per file).
5. **`MEMORY.md` memory index** (auto-loaded) — research history, prior findings, retractions; skim before
   concluding on baselines/shuffle/panel questions.

Slurm note: `sbatch`/`squeue` are NOT on the assistant's non-interactive PATH — the user runs Slurm commands
in their own terminal.
