"""
Evaluate RepoScout.

Two things can be evaluated:
  1. question answering about a codebase (default)
  2. a finished onboarding pack (--check-docs)

By default the questions are about the RepoScout code itself, so this works
out of the box - you know the right answers because you have the code. With
--fixtures it also runs against the tiny sample repos in evals/fixtures/
(Python/Flask, Node CLI, Go service), so results do not depend on one codebase.
To evaluate on your own project, write a questions file in the same format
and pass --repo and --questions.

Usage:
    python evals/run_evals.py --local                 # no LangSmith, prints a table
    python evals/run_evals.py                         # dataset + experiment in LangSmith
    python evals/run_evals.py --fixtures --local      # the sample repos in evals/fixtures/
    python evals/run_evals.py --judge --judge-model llama3.1:8b
    python evals/run_evals.py --repo ../my-app --questions my_questions.jsonl
    python evals/run_evals.py --check-docs output/docs --repo ../my-app

Takes:  a JSONL file, one object per line:
          {"question": "...", "keywords": ["..."], "files": ["..."]}
        A keyword may list alternatives with "|": "sqlite|sqlitesaver" passes if
        either appears (case-insensitive).
Gives:  per-question scores
          keyword_recall  share of expected keywords found in the answer
          cites_file      1 if the answer names at least one expected file
          paths_exist     share of file paths named in the answer that really exist
                          in the repo (a cheap hallucination check)
          judge           (optional) 1/0 from an LLM acting as grader
        In LangSmith mode you also get an experiment you can compare against
        earlier runs - useful after changing a prompt or the model.

The judge should be a DIFFERENT model than the one answering (--judge-model or
OLLAMA_JUDGE_MODEL); a small model grading its own answers is biased.

Every run costs one trace per question (plus the judge calls), well inside
the free 5k/month. The agent runs read-only here, so nothing is written to
your docs folder and there are no approval stops.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from reposcout.agent import build_agent  # noqa: E402
from reposcout.config import ConfigError, load_settings  # noqa: E402
from reposcout.llm import OllamaUnavailable, check_ollama, get_chat_model  # noqa: E402
from reposcout.validate import _candidate_paths, _exists, _repo_index, format_issues, validate_docs  # noqa: E402

DATASET_NAME = "reposcout-qa"
FIXTURES_DIR = HERE / "fixtures"


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_target(settings):
    agent = build_agent(settings, checkpointer=InMemorySaver(), store=InMemoryStore(), read_only=True)

    def answer(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 60}
        result = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]}, config)
        return {"answer": result["messages"][-1].text}

    return answer


# --- evaluators --------------------------------------------------------------
# LangSmith matches arguments by name: outputs = what the target returned,
# reference_outputs = the expected values stored with the example.

def keyword_found(keyword: str, answer: str) -> bool:
    """`a|b` means either alternative counts."""
    return any(alt.strip().lower() in answer for alt in keyword.split("|") if alt.strip())


def keyword_recall(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer", "").lower()
    keywords = reference_outputs.get("keywords", [])
    found = [k for k in keywords if keyword_found(k, answer)]
    score = len(found) / len(keywords) if keywords else 1.0
    return {"key": "keyword_recall", "score": score, "comment": f"found {found}"}


def cites_file(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer", "").lower()
    files = reference_outputs.get("files", [])
    hit = any(f.lower() in answer for f in files)
    return {"key": "cites_file", "score": int(hit)}


def make_paths_exist(settings):
    """Share of file paths mentioned in `backticks` that exist in the repo."""
    index = _repo_index(settings.repo_path, settings.ignores)

    def paths_exist(outputs: dict) -> dict:
        paths = _candidate_paths(outputs.get("answer", ""))
        if not paths:
            return {"key": "paths_exist", "score": 1.0, "comment": "no paths mentioned"}
        missing = sorted(p for p in paths if not _exists(p, index))
        return {"key": "paths_exist", "score": 1 - len(missing) / len(paths), "comment": f"missing {missing}"}

    return paths_exist


def make_judge(settings):
    judge_name = settings.judge_model or settings.ollama_model
    if judge_name == settings.ollama_model:
        print("warning: the judge is the same model as the agent - set --judge-model / OLLAMA_JUDGE_MODEL "
              "to a different one for an unbiased score")
    model = get_chat_model(settings, model=judge_name)

    def judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        prompt = (
            "You are grading an answer about a codebase.\n"
            f"Question: {inputs['question']}\n"
            f"Facts a correct answer should contain: {reference_outputs.get('keywords')} "
            f"and it should point to: {reference_outputs.get('files')}\n"
            f"Answer to grade:\n{outputs.get('answer', '')}\n\n"
            "Reply with exactly one word: PASS if the answer is correct and grounded, otherwise FAIL."
        )
        verdict = model.invoke(prompt).text.strip().upper()
        return {"key": "judge", "score": int(verdict.startswith("PASS")), "comment": verdict[:50]}

    return judge


# --- runners -------------------------------------------------------------------

def run_local(title: str, questions: list[dict], target, evaluators) -> dict[str, float]:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"RepoScout eval (local) - {title}")
    table.add_column("#")
    table.add_column("question", max_width=50)
    names = [e.__name__ for e in evaluators]
    for name in names:
        table.add_column(name)

    totals = {name: 0.0 for name in names}
    for i, q in enumerate(questions, 1):
        console.print(f"[dim]{i}/{len(questions)} {q['question']}[/]")
        inputs = {"question": q["question"]}
        reference = {"keywords": q.get("keywords", []), "files": q.get("files", [])}
        outputs = target(inputs)

        row = []
        for ev, name in zip(evaluators, names):
            if name == "judge":
                score = ev(inputs, outputs, reference)["score"]
            elif name == "paths_exist":
                score = ev(outputs)["score"]
            else:
                score = ev(outputs, reference)["score"]
            totals[name] += score
            row.append(f"{score:.2f}")
        table.add_row(str(i), q["question"], *row)

    averages = {n: totals[n] / len(questions) for n in names}
    table.add_row("", "[bold]average[/]", *[f"[bold]{averages[n]:.2f}[/]" for n in names])
    console.print(table)
    return averages


def sync_dataset(client, dataset_name: str, questions: list[dict]) -> None:
    """Create the dataset the first time; later runs reuse it."""
    if client.has_dataset(dataset_name=dataset_name):
        return
    dataset = client.create_dataset(dataset_name, description="Questions about a codebase for RepoScout")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": q["question"]},
                "outputs": {"keywords": q.get("keywords", []), "files": q.get("files", [])},
            }
            for q in questions
        ],
    )
    print(f"created dataset '{dataset_name}' with {len(questions)} examples")


def run_langsmith(dataset_name: str, questions: list[dict], target, evaluators, model_name: str) -> None:
    import os

    from langsmith import Client, evaluate

    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set. Add it to .env or use --local.")
        raise SystemExit(1)

    client = Client()
    sync_dataset(client, dataset_name, questions)
    evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"reposcout-{model_name.replace(':', '-')}",
        metadata={"model": model_name},
        max_concurrency=1,  # one local model, one request at a time
        client=client,
    )
    print(f"done - open the dataset '{dataset_name}' in LangSmith to see and compare experiments")


def check_docs(settings, docs: str) -> None:
    """Validate a finished pack: files exist, paths exist, mermaid looks valid."""
    issues = validate_docs(Path(docs).expanduser().resolve(), settings.repo_path, settings.ignores)
    if issues:
        print(f"{len(issues)} problem(s) in {docs}:\n{format_issues(issues)}")
        raise SystemExit(1)
    print(f"pack in {docs} is clean")


def one_run(args, settings, questions: list[dict], title: str, dataset_name: str):
    target = make_target(settings)
    evaluators = [keyword_recall, cites_file, make_paths_exist(settings)]
    if args.judge:
        evaluators.append(make_judge(settings))
    if args.local:
        return run_local(title, questions, target, evaluators)
    return run_langsmith(dataset_name, questions, target, evaluators, settings.ollama_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RepoScout")
    parser.add_argument("--local", action="store_true", help="run without LangSmith")
    parser.add_argument("--judge", action="store_true", help="add an LLM-as-judge score")
    parser.add_argument("--judge-model", help="Ollama model for the judge (default: OLLAMA_JUDGE_MODEL)")
    parser.add_argument("--repo", help="repo to ask about (default: REPOSCOUT_REPO_PATH)")
    parser.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    parser.add_argument("--fixtures", action="store_true", help="also run the sample repos in evals/fixtures/")
    parser.add_argument("--check-docs", metavar="DIR", help="validate a finished onboarding pack and exit")
    parser.add_argument("--limit", type=int, help="only run the first N questions per repo")
    parser.add_argument("--no-check", action="store_true", help="skip the Ollama availability check")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
    if args.repo:
        settings = settings.with_repo(args.repo)
    if args.judge_model:
        from dataclasses import replace
        settings = replace(settings, judge_model=args.judge_model)

    if args.check_docs:
        check_docs(settings, args.check_docs)
        return

    if not args.no_check:
        try:
            check_ollama(settings)
            if args.judge and settings.judge_model:
                check_ollama(replace_models(settings, settings.judge_model))
        except OllamaUnavailable as exc:
            raise SystemExit(str(exc)) from None

    runs = []
    if not args.fixtures or args.repo or args.questions != str(HERE / "questions.jsonl"):
        runs.append(("main", settings, Path(args.questions), DATASET_NAME))
    if args.fixtures:
        for fixture in sorted(p for p in FIXTURES_DIR.iterdir() if (p / "questions.jsonl").is_file()):
            runs.append((fixture.name, settings.with_repo(fixture), fixture / "questions.jsonl",
                         f"{DATASET_NAME}-{fixture.name}"))

    summary = {}
    for title, run_settings, questions_path, dataset in runs:
        questions = load_questions(questions_path)
        if args.limit:
            questions = questions[: args.limit]
        summary[title] = one_run(args, run_settings, questions, title, dataset)

    if args.local and len(summary) > 1:
        print("\nsummary (averages):")
        for title, averages in summary.items():
            print(f"  {title:12} " + "  ".join(f"{k}={v:.2f}" for k, v in averages.items()))


def replace_models(settings, model: str):
    """Settings whose only configured model is `model` (for the availability check)."""
    from dataclasses import replace
    return replace(settings, ollama_model=model, analyst_model="", writer_model="")


if __name__ == "__main__":
    main()
