---
name: theory-code-reviewer
description: >
  Use this skill whenever the user wants to check whether their code is consistent with the theory
  or methods paper. Triggers include: "check the code against the paper", "are there any inconsistencies",
  "does the code match the theory", "review the implementation against the methods", "align the code with
  the paper", or any request to audit, validate, or reconcile existing Python code against a methods document.
  Also trigger when the user asks whether a specific function or pipeline step matches what is described
  in the paper.
---

# Theory-Code Reviewer Skill

Reads the project's methods PDF and selected Python files, then walks through the code one function at a time — discussing how each relates to the theory and flagging divergences or missing steps. Changes are only made after explicit user approval.

---

## Step 1: Locate the methods PDF

Scan the project root for a file matching `*_methods.pdf`.

- If exactly one match is found, proceed with it.
- If multiple matches are found, list them and ask the user which one to use.
- If no match is found, tell the user: *"No `*_methods.pdf` file found in the project root. Please check the filename."*

---

## Step 2: Ask which files to review

Ask the user:

> *"Which Python files should I review? Please list the file paths."*

Wait for the user to respond before proceeding. Read all listed files in full before starting the review.

---

## Step 3: Build a theory map

Before reviewing any code, read the methods PDF and extract a concise internal map of the algorithm:

- The ordered sequence of algorithm steps as described in the paper
- Key inputs and outputs at each step
- Any explicit constraints, assumptions, or parameter definitions

Do not show this map to the user — use it internally as the reference for all subsequent comparisons.

---

## Step 4: Review one function at a time

Work through the functions in the order they appear in the file(s). For each function:

### 4a. Theory linkage
State which part of the methods paper this function corresponds to. Be specific — reference the algorithm step, section, or equation by name if possible. If the function has no clear counterpart in the paper, flag it explicitly:
> *"⚠️ No clear counterpart found in the methods paper for `function_name`."*

### 4b. Consistency check
Assess the function against the theory it corresponds to. Focus on:

- **Logic divergence** — does the implementation differ from what the paper describes? Flag specific lines or operations that don't match.
- **Missing steps** — are there steps described in the paper that are absent from this function or not implemented anywhere in the reviewed files?

If the function is fully consistent with the theory, say so explicitly — don't invent issues.

### 4c. Proposed changes (if any)
If issues were found, describe the change that should be made in plain language — not code yet. Be specific about what needs to change and why, referencing the paper.

### 4d. Pause for approval
End each function review with:

> *"Should I make these changes, or would you like to discuss first? Type 'ok' to apply, or let me know your thoughts."*

- If the user approves: apply the minimal, surgical change to the function. Do not refactor beyond what is needed to align with the theory.
- If the user wants to discuss: engage, revise the proposal if needed, and ask again before touching the code.
- If no changes are needed: confirm and move on with *"No changes needed — moving to the next function."*

---

## Step 5: End-of-review summary

After all functions have been reviewed, produce a brief summary:

- **Functions reviewed:** total count
- **Changes applied:** list of functions modified and what was changed
- **Outstanding gaps:** any missing algorithm steps that were flagged but not yet implemented (i.e. not covered by any function in the reviewed files)
- **No-change functions:** count of functions that were fully consistent with the theory

---

## Constraints

- **Never modify code without explicit user approval** — not even trivial fixes like renaming a variable.
- **Stay within the reviewed files** — do not modify files the user did not list.
- **Minimal changes only** — when applying a fix, change only what is necessary to align with the theory. Do not rewrite, refactor, or improve code beyond the scope of the inconsistency.
- **Do not invent theory** — if the paper is ambiguous or silent on a particular implementation detail, flag the ambiguity rather than assuming an interpretation.
