# parsimmon -- LLM-Guided Setup

Copy this to your coding agent (Claude Code, Codex, etc.):

```
Fetch and follow the instructions at <REPO_URL>/blob/main/llm_install.md
```

---

## Instructions

You are helping a user integrate parsimmon into their simulation project.
Follow these steps in order. Do not skip steps. Ask the user when decisions
are needed.

---

### Step 1: Determine install method

Ask the user how they want to install parsimmon:

- **pip + git** (recommended): `pip install git+<REPO_URL>.git`
- **Clone locally**: `git clone <REPO_URL>` and add to `sys.path` or install with `pip install -e .`

Before installing, check whether the user is inside a virtual environment:

- Run `python -c "import sys; print(sys.prefix, sys.base_prefix)"` and compare the two values.
- If they match, the user is on a system Python. Do not install without explicit confirmation.
  Warn them: "You are not in a virtual environment. Installing into system Python can break other
  tools. Options: create and activate a venv first, or proceed anyway (not recommended)."
- If they don't match, proceed with the install.

Install parsimmon using the chosen method. Confirm the install succeeded by running:

```
python -c "from parsimmon import ParameterSetManager; print('ok')"
```

If the import fails, surface the error and resolve it before continuing.

---

### Step 2: Verify git state

Run `git status` in the user's project root.

**If the project is a git repository and the working tree is dirty** (modified files,
staged changes, or untracked files that look significant):

- Stop. Do not modify any files yet.
- Show the user the status output and ask them to choose one of:
  - "Let me handle it" -- pause here; the user will clean up manually. Re-run `git status`
    after they say they are done, and repeat this check until the tree is clean.
  - "Stash my changes" -- run `git stash` and confirm with `git status`.
  - "Commit current changes" -- ask the user for a commit message, then run
    `git add -A && git commit -m "<their message>"`.
- After any of the above actions, re-run `git status`. Do not proceed until the working
  tree is clean (or the user has explicitly acknowledged they want to proceed dirty).

**If the directory is not a git repository**:

- Warn the user: "This project is not tracked by git. Without version control, changes
  made during setup cannot be easily reviewed or reverted."
- Ask whether to: run `git init && git add -A && git commit -m "initial commit"` to
  initialize tracking, or proceed without version control.

---

### Step 3: Evaluate project structure

Scan the project. Look for:

- How simulations are defined: simulation setup functions, parameter dictionaries,
  config files, dataclasses, or similar structures.
- How simulations are executed: runner scripts, notebooks, CLI entry points, `if __name__ == '__main__'` blocks.
- Where parameters live: inline constants, dedicated config modules, scattered across
  multiple files.
- Where analysis and plotting happen: post-run scripts, functions that consume results,
  figure-generation code.
- What simulation framework is in use, if any (e.g., a custom loop, a third-party
  library, or framework-agnostic code).

Summarize your findings to the user:

- Which files contain simulation logic.
- Which files contain parameters or configuration.
- Which files contain analysis or plotting.
- Any ambiguities or questions about structure.

Ask clarifying questions before continuing if the structure is unclear or the boundaries
between parameter definition, simulation logic, and analysis are not obvious.

---

### Step 4: Plan the separation

**Core principle:** parameters and analysis must be separate from simulation
creation and execution logic.

- **Simulation logic** is the function that takes parameters and produces results.
  In parsimmon, this is the `fn` argument passed to `pm.run(fn)`. Its signature is:
  ```python
  def fn(pars: dict, metadata: dict) -> result:
      ...
  ```
  `pars` is the fully resolved parameter dictionary for one simulation point.
  `metadata` contains `parameter_set`, `group`, `sim_id`, `group_id`, and `label`.

- **Parameters** are defined using the `@pm.add` decorator on builder functions that
  return a `ParameterSet` or a plain `dict`. Groups of override parameters are added
  with `ps.add('GroupName', {...})`. Ranges for Cartesian expansion are created with
  `ParameterSet.arange(...)`, `.linspace(...)`, `.logspace(...)`, or `.iter(...)`.

- **Analysis** functions are registered with `@pm.analysis('ParameterSetName')` and
  receive a `SimResult` scoped to their parameter set after `pm.run()` completes.

If your coding agent supports a plan mode or task-planning interface, switch to it now.

Produce a written plan that covers:

1. Which code is simulation logic vs. parameter definition vs. analysis.
2. Where each piece should live after the refactor. Adapt to the user's existing
   directory structure. Do not impose a new layout unless the user's project has no
   clear structure.
3. How to wrap the existing simulation logic in a function with the signature
   `fn(pars: dict, metadata: dict) -> result`.
4. How to construct a `ParameterSetManager` and define parameter sets with `@pm.add`
   that reproduce the user's current parameter configurations.
5. Whether to enable caching. Recommend `cache=True` if the simulations are expensive
   and deterministic. With caching, results are stored in `data/cache/` by default and
   shared across parameter sets that overlap in parameter values.
6. Any other notable changes (file splits, renamed variables, imports to add or remove).

Present the plan to the user and get explicit approval before making any changes.

---

### Step 5: Implement

Execute the approved plan:

1. Make the changes described in the plan. Work file by file; do not scatter unrelated
   edits across a large diff.
2. After all changes are made, run any existing tests in the project:
   ```
   python -m pytest      # or the project's test command
   ```
   If tests fail due to the integration, fix them before proceeding.
3. Run a minimal end-to-end example to confirm parsimmon is working:
   ```python
   # example -- adapt to the user's actual sim function and parameters
    from parsimmon import ParameterSetManager, ParameterSet

   pm = ParameterSetManager()

   @pm.add
   def baseline():
       ps = ParameterSet()
       ps.add('Default', {'param': 1.0})
       return ps

   def run_sim(pars: dict, metadata: dict):
       # replace with a call to the user's actual simulation logic
       return {'result': pars['param'] * 2}

   results = pm.run(run_sim)
   print(results)
   for r in results:
       print(r)
   ```
4. Show the user the output and confirm the integration is working as expected.

For full documentation on `ParameterSet`, `ParameterSetManager`, `SimResult`, caching,
and custom backends, see `README.md` in the parsimmon repository.
