# RepoScout

A local codebase onboarding agent. Point it at any folder on your machine and it writes an onboarding pack for that code: You point it at any folder on your laptop, such as one of your own old projects or a cloned open-source repo. It works out how the code fits together and writes an onboarding pack for it: an architecture overview, a summary of each module, a “how a request flows through the system” walkthrough, and a list of risky or unclear areas.

| File | What's in it |
|---|---|
| `OVERVIEW.md` | what the project does, tech stack, how to run it, a component diagram |
| `MODULES.md` | one section per important package: purpose, key files, dependencies, gotchas |
| `FLOW.md` | one real request/command traced from entry point to where the work happens, with a sequence diagram |
| `RISKS.md` | fragile areas, missing tests, unclear code |

Every file is shown to you before it's written, and you can approve it or send it back with feedback. The code being studied is mounted read only, so the agent can't change it.

Runs fully on your laptop with Ollama. LangSmith is optional and only used for traces and evals.

---

## What this project teaches

| Concept | Where to look |
|---|---|
| Planning with `write_todos` | `TodoListMiddleware` in `agent.py`, step 1 of `MAIN_PROMPT` |
| A virtual filesystem made of several mounts | `build_backend()` in `agent.py` |
| Read-only mounts and approval gates | `build_permissions()` in `agent.py` |
| Subagents with their own prompt and tools | the `subagents` list in `agent.py` |
| Human in the loop: approve / reject with feedback | `console.py` (`run_turn`, `_ask_decision`) |
| Long-term memory that survives restarts | `/memories/AGENTS.md` -> `StoreBackend` + SQLite store |
| Resumable sessions | SQLite checkpointer in `cli.py`, `--session` flag |
| Custom tools on top of the built-in ones | `tools.py` |
| Tracing and evaluation | LangSmith env vars, `evals/run_evals.py` |


## Project layout

```
reposcout/
├── reposcout/
│   ├── __init__.py     package overview
│   ├── config.py       reads .env into a Settings object
│   ├── llm.py          ChatOllama with a sane context size + Ollama check
│   ├── fsutil.py       pruned file walking, secret-file rules
│   ├── tools.py        repo_map, outlines, entry points, import graph, risk tools, module cache
│   ├── validate.py     checks the generated pack (paths exist, mermaid valid)
│   ├── sessions.py     ties a chat session to the repo it started with
│   ├── prompts.py      main + subagent system prompts
│   ├── agent.py        backend, permissions, subagents -> create_deep_agent
│   ├── graph.py        entry for `langgraph dev` / Studio
│   ├── console.py      terminal output + approve/reject prompts
│   └── cli.py          python -m reposcout.cli
├── tests/              pytest suite (tools, permissions, validator, config)
├── evals/
│   ├── questions.jsonl sample questions (about RepoScout itself)
│   ├── fixtures/       three tiny sample repos (Flask, Node, Go) with their own questions
│   └── run_evals.py    local or LangSmith evaluation, plus --check-docs
├── docs/               architecture and run-flow PDFs
├── pyproject.toml      packaging + the `reposcout` command
├── constraints.txt     exact tested dependency versions
├── langgraph.json      config for `langgraph dev`
├── requirements.txt
└── .env.example
```

---

## Setup

You need:

- **Python 3.11 or newer** (deepagents requires it)
- **Ollama** - https://ollama.com/download
- About 5 GB free disk for the model, 8 GB+ RAM

### 1. Get the model (both OS)

```bash
ollama pull qwen2.5:7b
ollama list          # check it's there
```

Ollama runs as a background app after install. If `ollama list` can't connect, start it with `ollama serve` in a separate terminal.

### 2a. macOS

```bash
cd reposcout
python3 -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -e ".[dev]"
cp .env.example .env
```

### 2b. Windows (PowerShell)

```powershell
cd reposcout
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -c constraints.txt -e ".[dev]"
Copy-Item .env.example .env
```

If PowerShell says running scripts is disabled, run this once and open a new terminal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Using Command Prompt instead of PowerShell? Activate with `.venv\Scripts\activate.bat`.

### 3. Run it

Study RepoScout itself first (no flags needed - it defaults to its own folder):/

```bash
python -m reposcout.cli --pack
```

Then try a real project:

```bash
# macOS
python -m reposcout.cli --repo ~/code/my-project --pack

# Windows
python -m reposcout.cli --repo C:\Users\you\code\my-project --pack
```

Or just chat about the code without building docs:

```bash
python -m reposcout.cli --repo ../some-project
you: where is authentication handled?
you: I prefer short docs with bullet points, remember that
you: /memory
```

The pack ends up in `output/docs/` (change it with `--docs` or `REPOSCOUT_DOCS_DIR`).

Each run prints its session name. Continue it later with `--session scout-1a2b3c4d`. A session is tied to the repo it started with; resuming it with a different `--repo` is refused so old context never mixes with new code.

After `--pack` the pack is checked automatically: all four files present, every `path` in backticks exists in the repo, and mermaid blocks parse. If something is wrong you are offered one fix-up round. After `pip install -e .` you can also run `reposcout --repo ...` instead of `python -m reposcout.cli`.

Ctrl-C while the agent works cancels that turn; Ctrl-C at the prompt exits.

### What the agent can and cannot do

- `/repo/` is read only, and credential files (`.env*`, `*.pem`, `id_rsa`, `credentials.json`, ... but not `.env.example`) cannot be read at all and are hidden from the tools.
- Writes to `/docs/` **and** `/memories/` need your approval. Text in the studied repo is untrusted, so a poisoned file cannot silently change what the agent remembers.
- Module notes are cached in `.reposcout/cache/<repo>/modules/` under a fingerprint of the folder. A second `--pack` only re-analyses folders that changed.

### Models per role (optional)

Set `OLLAMA_ANALYST_MODEL` (module-analyst, flow-tracer, risk-analyst) and/or `OLLAMA_WRITER_MODEL` (doc-writer) in `.env` to use different models; empty means `OLLAMA_MODEL`.

### Tests

```bash
python -m pytest -q        # no Ollama needed
```

---

## LangGraph Studio (optional, free, local)

Studio lets you watch the graph run, inspect the state and the virtual files, and click through the approval interrupts.

```bash
pip install "langgraph-cli[inmem]"
langgraph dev
```

It opens Studio in your browser. Set `REPOSCOUT_REPO_PATH` in `.env` to choose which repo to study, then send a message like *"Build the full onboarding pack"*. When the agent hits an approval gate, Studio shows the interrupt and you resume it from there.

`langgraph dev` runs on your machine. It asks you to sign in to LangSmith for the Studio UI, which the free plan covers.

---

## LangSmith tracing (optional, free)

1. Sign up at https://smith.langchain.com - the Developer plan is free. **Don't add a credit card.** Without one, your account is hard-capped at 5,000 traces a month, so it can never bill you.
2. Settings -> API Keys -> create a key.
3. In `.env`:
   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=reposcout
   ```
4. Run the CLI again. Each turn shows up as one trace, with every subagent and tool call nested inside it.

Turn `LANGSMITH_TRACING=false` while you're doing throwaway runs to save quota.

Things worth looking at in a trace:

- how many tokens the main agent spends vs. the subagents (this is why subagents exist - context isolation)
- the interrupt step and what happened after a rejection
- calls that ended in `permission denied` - the model trying to write to `/repo/`

---

## Evals

```bash
python evals/run_evals.py --local            # no account needed
python evals/run_evals.py --local --judge --judge-model llama3.1:8b   # LLM judge, a DIFFERENT model
python evals/run_evals.py --local --fixtures # also run the Flask / Node / Go sample repos
python evals/run_evals.py                    # push dataset + experiment to LangSmith
python evals/run_evals.py --check-docs output/docs --repo ../my-app   # validate a finished pack
```

The default questions are about RepoScout's own code, so you can check the answers yourself. Scores:

- `keyword_recall` - share of expected facts mentioned in the answer
- `cites_file` - did it point at the right file
- `paths_exist` - share of file paths named in the answer that really exist in the repo (hallucination check)
- `judge` - a model grades the answer PASS/FAIL. Use a different model than the one answering (`--judge-model` / `OLLAMA_JUDGE_MODEL`); a small model grading itself is biased.

A keyword can list alternatives with `|`, e.g. `"sqlite|sqlitesaver"`.

For your own repo, copy `evals/questions.jsonl`, write 10-15 questions you know the answers to, and run:

```bash
python evals/run_evals.py --local --repo ../my-app --questions my_questions.jsonl
```

A good loop: run the evals, change one prompt in `prompts.py`, run them again, compare. In LangSmith, open the `reposcout-qa` dataset to see experiments side by side.

---

## Things to try

1. Run `--pack` once with subagents and once after deleting them from the `subagents` list. Compare tokens in LangSmith.
2. Change `/docs/**` from `mode="interrupt"` to `mode="allow"` and see how the run feels without reviews.
3. Reject a file with specific feedback and check the rewrite actually used it.
4. Tell it a preference ("always include a 'how to run tests' section"), quit, start a new session and check it remembers (`/memory`).
5. Try a bigger model (`qwen2.5:14b`) if your machine can take it, and compare eval scores.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 11434 | Ollama isn't running - open the app or run `ollama serve` |
| The model answers without using tools / ignores the plan | Context too small. Keep `OLLAMA_NUM_CTX=16384` or raise it. Also try a larger model. |
| `model "qwen2.5:7b" not found` | `ollama pull qwen2.5:7b` |
| Very slow | Normal on CPU-only machines. Use `--repo` on a small project first, or try `qwen2.5:3b` (weaker at tool calling). |
| `Cannot reach Ollama` / `Model(s) not installed` on start | The CLI checks Ollama first and prints the exact `ollama pull` command. Skip the check with `--no-check`. |
| Run stops with a recursion error | Big repo. Raise `recursion_limit` in `cli.py`, or point `--repo` at a subfolder. |
| `ModuleNotFoundError: reposcout` | Run commands from the `reposcout/` project folder with the venv active. |
| Windows path with spaces | Quote it: `--repo "C:\Users\you\My Projects\app"` |
