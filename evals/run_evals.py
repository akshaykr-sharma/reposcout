"""
Evaluate RepoScout's question answering.

By default the questions are about the RepoScout code itself, so this works
out of the box - you know the right answers because you have the code. To
evaluate on your own project, write a questions file in the same format and
pass --repo and --questions.

Usage:
    python evals/run_evals.py --local               # no LangSmith, prints a table
    python evals/run_evals.py                       # dataset + experiment in LangSmith
    python evals/run_evals.py --judge               # also score with the local LLM as judge
    python evals/run_evals.py --repo ../my-app --questions my_questions.jsonl

Takes:  a JSONL file, one object per line:
          {"question": "...", "keywords": ["..."], "files": ["..."]}
Gives:  per-question scores
          keyword_recall  share of expected keywords found in the answer
          cites_file      1 if the answer names at least one expected file
          judge           (optional) 1/0 from the local model acting as grader
        In LangSmith mode you also get an experiment you can compare against
        earlier runs - useful after changing a prompt or the model.

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
from reposcout.config import load_settings  # noqa: E402
from reposcout.llm import get_chat_model  # noqa: E402

DATASET_NAME = "reposcout-qa"


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_target(repo: str | None):
    settings = load_settings()
    if repo:
        settings = settings.with_repo(repo)
    agent = build_agent(settings, checkpointer=InMemorySaver(), store=InMemoryStore(), read_only=True)

    def answer(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"}, "recursion_limit": 60}
        result = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]}, config)
        return {"answer": result["messages"][-1].text}

    return answer, settings


# --- evaluators --------------------------------------------------------------
# LangSmith matches arguments by name: outputs = what the target returned,
# reference_outputs = the expected values stored with the example.

def keyword_recall(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer", "").lower()
    keywords = reference_outputs.get("keywords", [])
    found = [k for k in keywords if k.lower() in answer]
    score = len(found) / len(keywords) if keywords else 1.0
    return {"key": "keyword_recall", "score": score, "comment": f"found {found}"}


def cites_file(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer", "").lower()
    files = reference_outputs.get("files", [])
    hit = any(f.lower() in answer for f in files)
    return {"key": "cites_file", "score": int(hit)}


def make_judge(settings):
    model = get_chat_model(settings)

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

def run_local(questions: list[dict], target, evaluators) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="RepoScout eval (local)")
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
            else:
                score = ev(outputs, reference)["score"]
            totals[name] += score
            row.append(f"{score:.2f}")
        table.add_row(str(i), q["question"], *row)

    table.add_row("", "[bold]average[/]", *[f"[bold]{totals[n] / len(questions):.2f}[/]" for n in names])
    console.print(table)


def sync_dataset(client, questions: list[dict]) -> None:
    """Create the dataset the first time; later runs reuse it."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(DATASET_NAME, description="Questions about a codebase for RepoScout")
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
    print(f"created dataset '{DATASET_NAME}' with {len(questions)} examples")


def run_langsmith(questions: list[dict], target, evaluators, model_name: str) -> None:
    import os

    from langsmith import Client, evaluate

    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set. Add it to .env or use --local.")
        raise SystemExit(1)

    client = Client()
    sync_dataset(client, questions)
    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=f"reposcout-{model_name.replace(':', '-')}",
        metadata={"model": model_name},
        max_concurrency=1,  # one local model, one request at a time
        client=client,
    )
    print("done - open the dataset in LangSmith to see and compare experiments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RepoScout")
    parser.add_argument("--local", action="store_true", help="run without LangSmith")
    parser.add_argument("--judge", action="store_true", help="add an LLM-as-judge score")
    parser.add_argument("--repo", help="repo to ask about (default: REPOSCOUT_REPO_PATH)")
    parser.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    target, settings = make_target(args.repo)
    evaluators = [keyword_recall, cites_file]
    if args.judge:
        evaluators.append(make_judge(settings))

    if args.local:
        run_local(questions, target, evaluators)
    else:
        run_langsmith(questions, target, evaluators, settings.ollama_model)


if __name__ == "__main__":
    main()
