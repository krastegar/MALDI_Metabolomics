---
name: scientific-reviewer
description: >
  Use this skill whenever the user wants to review, summarize, or plan implementation from a methods paper.
  Triggers include: "review the methods paper", "summarize the PDF", "read the paper", "plan the implementation",
  "what does the methods paper say", "summarize the project scope", or any request that involves understanding
  or acting on the project's methods document. Also trigger automatically when a task involves implementing
  an algorithm and a *_methods.pdf file is present in the project root — read it before writing any code.
---

# Scientific Reviewer Skill

Reads the project's methods PDF, produces a structured scientific summary, and — with user approval — hands off to the chunked-python-pipeline skill for implementation.

---

## Step 1: Locate the methods PDF

Scan the project root directory for a file matching the pattern `*_methods.pdf`. 

- If exactly one match is found, proceed with it.
- If multiple matches are found, list them and ask the user which one to use.
- If no match is found, tell the user: *"No `*_methods.pdf` file found in the project root. Please check the filename."*

---

## Step 2: Read and summarize the document

Read the full PDF and produce a **structured scientific summary**. Keep it concise — no longer than 3 pages of prose. Use the following structure:

### Summary structure

**1. Background & Motivation**
- What problem does this paper address?
- Why does it matter? What gap does it fill?

**2. Objectives**
- What are the explicit goals of the method/paper?
- What are the primary outputs or deliverables?

**3. Data & Inputs**
- What data does the method operate on?
- Key properties: modality, format, dimensionality, any preprocessing assumptions.

**4. Algorithm & Methods**
- Core algorithmic steps, in order.
- Key mathematical formulations (state them clearly but briefly — no full derivations).
- Any model assumptions or constraints worth flagging.

**5. Evaluation & Validation**
- How is the method evaluated?
- Metrics, baselines, datasets used for validation.

**6. Implementation-Relevant Notes**
- Dependencies, libraries, or computational requirements mentioned in the paper.
- Any known edge cases, failure modes, or caveats the authors flag.

---

## Step 3: Pause for approval

After presenting the summary, ask the user:

> *"Does this summary look accurate? Anything to correct or add before I proceed?"*

Do not proceed until the user confirms. If they request corrections, update the relevant section(s) and confirm again.

---

## Step 4: Propose an implementation plan

Based on the confirmed summary, propose a high-level implementation plan — a numbered list of pipeline stages derived directly from the algorithm steps in the methods paper. For each stage, include:

- A short name (e.g. `load_data`, `compute_tmap`)
- One sentence describing what it does

Keep it at the pipeline level — this is not pseudocode, just a clear sequence of stages the user can reason about.

Then ask:

> *"Does this plan look right? Any stages to add, remove, or reorder before I start writing code?"*

Do not proceed until the user approves the plan. If they request changes, revise and confirm again.

---

## Step 5: Hand off to chunked-python-pipeline

Once the plan is approved, invoke the **chunked-python-pipeline** skill to begin implementation. Pass the following context:

- The confirmed summary (objectives, inputs, algorithm steps, implementation notes)
- The approved implementation plan as the starting chunk outline

The chunked-python-pipeline skill takes it from here — it will begin writing chunks in order, pausing for approval after each one.

---

## Tone and format

- Write the summary in clear scientific prose — not bullet-point fragments, but not padded either.
- Use section headers exactly as listed above.
- **Always render LaTeX** — never leave raw LaTeX strings in the output (e.g. no `\hat{\mu}_z` or `\sum_{i=1}^{N}`). All mathematical expressions must be rendered using markdown math syntax: inline with `$...$` and display equations with `$$...$$`. For example, write $\hat{\mu}_z$ not `\hat{\mu}_z`, and use display blocks for key equations: $$p(z \mid x) \propto \mathcal{N}(x; \mu_z, \Sigma)$$
- Flag anything ambiguous or underspecified in the paper with a brief note, e.g.: *"⚠️ The paper does not specify how missing values are handled prior to normalization."*
- Do not invent details not present in the paper.