---
name: chunked-python-pipeline
description: >
  Use this skill whenever the user wants to build a Python pipeline, script, or multi-step workflow
  and wants to review it incrementally before seeing the full thing. Triggers include: "build a pipeline",
  "write a script step by step", "show me the code in chunks", "let me review as we go", "write it piece by piece",
  or any request to write a Python pipeline or script where the user may want control over the process.
  Also trigger when the user is building preprocessing, analysis, or modeling workflows in Python and
  hasn't explicitly asked for the full script at once — lean toward chunked delivery over dumping everything at once.
---

# Chunked Python Pipeline Skill

A workflow for building Python pipelines incrementally — one logical chunk at a time — so the user can review and approve before proceeding. Ends with a full-pipeline correctness review.

---

## When to use this skill

- User wants to build a Python script or pipeline
- User wants to review code as it's written, not after the fact
- User is building a multi-step workflow (preprocessing, modeling, analysis, etc.)

---

## Workflow

### Phase 1: Plan before writing

Before writing any code:

1. **Confirm the goal** — briefly restate what the pipeline should do in 1–2 sentences and ask the user to confirm.
2. **Lay out the chunk plan** — list all planned chunks by name/purpose (e.g., `1. load_data`, `2. filter_missing`, `3. normalize`, …). This gives the user an overview and lets them flag structural issues early.
3. Wait for the user to confirm the plan before writing chunk 1.

---

### Phase 2: Write chunks one at a time

For each chunk:

1. **Preview** — one sentence describing what the next chunk will do. Example:
   > *"Next: `normalize_columns()` — applies PQN normalization per sample and caps outliers."*

2. **Write the chunk** — follow the constraints below.

3. **Pause** — end with a short prompt like:
   > *"Ready for the next chunk? (`load_metadata` — reads the per-mouse metadata CSV)"*
   
   Do not proceed until the user confirms (e.g., "ok", "next", "go ahead", "looks good").

---

### Chunk writing constraints

- **~20 lines of executable code** as a soft cap — only lines that actually execute count (imports, assignments, function calls, control flow). Blank lines, comments, and docstrings do not count. Never break a logical unit mid-function to hit the line count; a cohesive logical unit that runs slightly over 20 executable lines is fine.
- **Verbose comments encouraged** — comment liberally to explain intent, non-obvious logic, and parameter choices. Comments do not count toward the line cap.
- **Docstrings with typed I/O** — every function gets a docstring that documents each input and output variable with its type and a one-line description of what it is. Use a `Args:`/`Returns:` style (Google/NumPy), e.g. `df (pd.DataFrame): per-pixel intensity matrix, rows = pixels`. Type the parameters and the return value even if the signature already carries type hints — the docstring is the interface contract. Docstrings do not count toward the line cap.
- **Lean toward functions** — most chunks should be a single, well-named function. Avoid classes unless the user explicitly requests them or the design clearly requires one.
- **No overengineering** — write the simplest code that correctly solves the problem. No unnecessary abstractions, base classes, factory patterns, config objects, or generalization beyond what's needed. If a plain variable works, use it; don't wrap it in a dataclass.
- **Prefer libraries over hand-rolled logic** — always reach for a well-established library before implementing an algorithm manually. Prefer libraries with C/C++ backends for performance: numpy, scipy, scikit-learn, scikit-image, pandas, opencv-python, etc. Only write pure Python logic when no suitable library function exists.
- **Imports at the top of the first chunk only** — subsequent chunks assume the imports are already present. If a new dependency is introduced mid-pipeline, note it explicitly: *"This chunk needs `from scipy import stats` — add it to your imports."*
- **Consistent naming** — use names established in earlier chunks. Don't rename variables or introduce synonyms.
- **No placeholder logic** — every chunk must be real, runnable code. No `pass`, `# TODO`, or stub functions.

---

### Phase 3: Full-pipeline review

After all chunks are approved, assemble the complete pipeline and perform a review focused on **logic flow and correctness**. Structure the review as follows:

#### 3a. Assembled pipeline
Show the full script, with chunks in order and imports at the top. No extra commentary here — just clean code.

#### 3b. Review notes
After the code block, write a short review covering:

- **Data flow** — does data pass correctly between functions? Are shapes/types compatible at each handoff?
- **Logic errors** — any off-by-one errors, incorrect conditions, wrong aggregation direction, etc.?
- **Silent failures** — anything that will run without error but produce wrong results (e.g., operating on a copy instead of the original, wrong axis in numpy/pandas operations)?
- **Missing edge case handling** — e.g., empty inputs, all-NaN columns, division by zero — flag only cases that are likely given the pipeline's context.

Format: a short bulleted list. If no issues are found, say so explicitly — don't invent issues.

#### 3c. Suggested fixes (if any)
If issues were found in 3b, show targeted diffs or replacement code snippets — not a full rewrite. Each fix should be minimal and surgical.

---

## Tone and pacing

- Be concise between chunks — the preview and pause lines should be short.
- Don't editorialize on each chunk ("great function!", "this is clean") — just write and move on.
- If the user asks to skip the plan step or review step, respect that.
- If the user wants to revise a chunk before continuing, do it in place before moving to the next one.
