"""Analyze the Rousseau Vector RAG vs GraphRAG LLM-as-a-judge benchmark.

Run from anywhere:
    python benchmark/analysis.py

Requires only the Python standard library. matplotlib is optional for charts.
"""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
EVALUATIONS_FILE = BENCHMARK_DIR / "evaluations.json"
QUESTIONS_FILE = BENCHMARK_DIR / "questions.json"
VECTOR_FILE = BENCHMARK_DIR / "vector_answers.json"
GRAPHRAG_FILE = BENCHMARK_DIR / "graphrag_answers.json"
RESULTS_DIR = BENCHMARK_DIR / "analysis_results"

SYSTEMS = ("graphrag", "vector_rag")
CRITERIA = ("comprehensiveness", "diversity", "empowerment", "directness")
CATEGORIES = (
    "local_fact_retrieval",
    "cross_document_reasoning",
    "global_sensemaking",
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------
# COLLAPSE REPEATED JUDGMENTS TO ONE VOTE PER QUESTION/METRIC
# ---------------------------------------------------------


def summarize_votes(votes, expected_runs):
    """Require the full number of runs; do not mistake incomplete data for a tie."""
    counts = Counter(votes)
    if len(votes) != expected_runs:
        return "incomplete", counts

    # Strict majority: with three runs, a result needs at least two votes.
    for outcome in ("graphrag", "vector_rag", "tie"):
        if counts[outcome] > expected_runs / 2:
            return outcome, counts

    # Example: one GraphRAG vote, one Vector RAG vote, one tie.
    return "no_majority", counts


def aggregate_judgments(data):
    expected_runs = int(data.get("judge_runs_per_criterion", 1))
    if expected_runs < 1:
        raise ValueError("judge_runs_per_criterion must be at least 1.")

    raw = data.get("evaluations", [])
    statuses = Counter(row.get("status", "unknown") for row in raw)

    # If evaluate.py retried an API error, a question/criterion/run can occur
    # more than once. Prefer the most recent SUCCESSFUL record, so one run
    # never contributes multiple votes.
    latest = {}
    for row in raw:
        key = (row["question_id"], row["criterion"], int(row["run_number"]))
        old = latest.get(key)
        if old is None or row.get("status") == "completed" or old.get("status") != "completed":
            latest[key] = row

    grouped = defaultdict(list)
    for (question_id, criterion, run_number), row in latest.items():
        if run_number < 1 or run_number > expected_runs:
            continue
        grouped[(question_id, criterion)].append(row)

    results = []
    for (question_id, criterion), records in sorted(grouped.items()):
        # A failed/skipped call is not a judgment. Only completed records count.
        completed = [
            record for record in records
            if record.get("status") == "completed"
            and record.get("winner_system") in (*SYSTEMS, "tie")
        ]
        votes = [record["winner_system"] for record in completed]
        decision, counts = summarize_votes(votes, expected_runs)
        source = records[0]

        results.append({
            "question_id": question_id,
            "category": source["category"],
            "criterion": criterion,
            "expected_runs": expected_runs,
            "completed_runs": len(completed),
            "graphrag_votes": counts["graphrag"],
            "vector_rag_votes": counts["vector_rag"],
            "tie_votes": counts["tie"],
            "majority_winner": decision,
        })

    return results, statuses, expected_runs


# ---------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ---------------------------------------------------------


def summarize_group(rows, label, group_type):
    usable = [row for row in rows if row["majority_winner"] != "incomplete"]
    counts = Counter(row["majority_winner"] for row in usable)
    total = len(usable)
    graph = counts["graphrag"]
    vector = counts["vector_rag"]
    tie = counts["tie"]
    no_majority = counts["no_majority"]

    # Consistent with the paper's 100/0/50 scheme: ties (including cases
    # without a majority) contribute half to each system's preference share.
    graph_share = (graph + 0.5 * (tie + no_majority)) / total if total else None
    vector_share = (vector + 0.5 * (tie + no_majority)) / total if total else None

    return {
        "group_type": group_type,
        "group": label,
        "evaluated_pairs": total,
        "incomplete_pairs": len(rows) - total,
        "graphrag_wins": graph,
        "vector_rag_wins": vector,
        "ties": tie,
        "no_majority": no_majority,
        "graphrag_win_pct": round(100 * graph / total, 2) if total else None,
        "vector_rag_win_pct": round(100 * vector / total, 2) if total else None,
        "tie_pct": round(100 * tie / total, 2) if total else None,
        "graphrag_preference_share_pct": round(100 * graph_share, 2) if total else None,
        "vector_rag_preference_share_pct": round(100 * vector_share, 2) if total else None,
    }


def exact_sign_test(graph_wins, vector_wins):
    """Two-sided exact sign test on decisive question-level comparisons."""
    n = graph_wins + vector_wins
    if n == 0:
        return None
    k = min(graph_wins, vector_wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def holm_adjust(pvalues):
    """Holm-adjust p-values across the four overall criterion tests."""
    valid = sorted((key, p) for key, p in pvalues.items() if p is not None)
    valid.sort(key=lambda item: item[1])
    adjusted = {key: None for key in pvalues}
    running_max = 0.0
    m = len(valid)
    for index, (key, pvalue) in enumerate(valid):
        running_max = max(running_max, (m - index) * pvalue)
        adjusted[key] = min(1.0, running_max)
    return adjusted


def summarize_latencies(path, system):
    if not path.exists():
        return {"system": system, "successful_questions": 0,
                "mean_seconds": None, "median_seconds": None}

    data = load_json(path)
    seconds = [
        float(row["latency_seconds"])
        for row in data.get("results", [])
        if row.get("answer") and not row.get("error")
        and isinstance(row.get("latency_seconds"), (int, float))
    ]
    return {
        "system": system,
        "successful_questions": len(seconds),
        "mean_seconds": round(statistics.mean(seconds), 3) if seconds else None,
        "median_seconds": round(statistics.median(seconds), 3) if seconds else None,
    }


# ---------------------------------------------------------
# OPTIONAL PLOT
# ---------------------------------------------------------


def save_chart(metric_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chart skipped (matplotlib is not installed). CSV/JSON results are still saved.")
        return

    valid = [row for row in metric_rows if row["evaluated_pairs"]]
    if not valid:
        return

    positions = list(range(len(valid)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([x - width / 2 for x in positions],
           [row["graphrag_win_pct"] for row in valid], width, label="GraphRAG")
    ax.bar([x + width / 2 for x in positions],
           [row["vector_rag_win_pct"] for row in valid], width, label="Vector RAG")
    ax.set_xticks(positions, [row["group"].title() for row in valid])
    ax.set_ylabel("Decisive wins / all evaluated pairs (%)")
    ax.set_title("LLM-as-a-Judge results by criterion")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metric_win_rates.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    data = load_json(EVALUATIONS_FILE)
    results, statuses, expected_runs = aggregate_judgments(data)
    RESULTS_DIR.mkdir(exist_ok=True)

    overall = summarize_group(results, "all_criteria", "overall")
    metric_rows = [
        summarize_group([row for row in results if row["criterion"] == criterion],
                        criterion, "criterion")
        for criterion in CRITERIA
    ]
    category_rows = [
        summarize_group([row for row in results if row["category"] == category],
                        category, "category")
        for category in CATEGORIES
    ]
    category_metric_rows = [
        summarize_group(
            [row for row in results
             if row["category"] == category and row["criterion"] == criterion],
            f"{category} / {criterion}", "category_criterion"
        )
        for category in CATEGORIES
        for criterion in CRITERIA
    ]

    # Test on independent QUESTION-level majority decisions, not on the
    # individual repeated judge calls. Exclude ties and no-majority outcomes.
    pvalues = {
        row["group"]: exact_sign_test(row["graphrag_wins"], row["vector_rag_wins"])
        for row in metric_rows
    }
    adjusted = holm_adjust(pvalues)
    for row in metric_rows:
        name = row["group"]
        row["sign_test_p"] = pvalues[name]
        row["holm_adjusted_p"] = adjusted[name]

    latencies = [
        summarize_latencies(VECTOR_FILE, "vector_rag"),
        summarize_latencies(GRAPHRAG_FILE, "graphrag"),
    ]

    columns = list(results[0]) if results else [
        "question_id", "category", "criterion", "expected_runs",
        "completed_runs", "graphrag_votes", "vector_rag_votes",
        "tie_votes", "majority_winner",
    ]
    summary_columns = list(metric_rows[0])
    write_csv(RESULTS_DIR / "question_results.csv", results, columns)
    write_csv(RESULTS_DIR / "metric_summary.csv", metric_rows, summary_columns)
    write_csv(RESULTS_DIR / "category_summary.csv", category_rows, list(category_rows[0]))
    write_csv(RESULTS_DIR / "category_metric_summary.csv", category_metric_rows,
              list(category_metric_rows[0]))
    write_csv(RESULTS_DIR / "latency_summary.csv", latencies, list(latencies[0]))

    expected_questions = None
    if QUESTIONS_FILE.exists():
        expected_questions = len(load_json(QUESTIONS_FILE).get("questions", []))

    unique_questions = len({row["question_id"] for row in results})
    report = {
        "benchmark_name": data.get("benchmark_name"),
        "judge_model": data.get("judge_model"),
        "expected_runs_per_criterion": expected_runs,
        "raw_record_statuses": dict(statuses),
        "expected_questions": expected_questions,
        "questions_with_any_judgments": unique_questions,
        "overall": overall,
        "by_criterion": metric_rows,
        "by_category": category_rows,
        "by_category_and_criterion": category_metric_rows,
        "latency": latencies,
        "notes": [
            "Each question/criterion contributes at most one majority decision.",
            "Incomplete groups are excluded from rate denominators.",
            "No-majority outcomes are kept separate from judge-declared ties; "
            "both count as 0.5 for preference share.",
            "Exact sign tests use decisive question-level decisions only; "
            "four overall criterion p-values use Holm adjustment.",
            "The judge compares answers, not their grounding in Rousseau source passages: "
            "these results are not faithfulness/hallucination scores.",
            "Latency includes subprocess startup if the GraphRAG runner invokes the CLI per question.",
        ],
    }
    write_json(RESULTS_DIR / "summary.json", report)
    save_chart(metric_rows)

    print("\nROUSSEAU BENCHMARK ANALYSIS")
    print(f"Judge model: {data.get('judge_model', 'unknown')}")
    print(f"Runs expected per question/criterion: {expected_runs}")
    print(f"Questions with any judgments: {unique_questions}",
          f"/ {expected_questions}" if expected_questions is not None else "")
    print(f"Completed comparison groups: {overall['evaluated_pairs']}")
    print(f"Incomplete comparison groups: {overall['incomplete_pairs']}")
    print("\nCriterion             GraphRAG  Vector RAG  Tie  No majority  Graph share")
    for row in metric_rows:
        share = row["graphrag_preference_share_pct"]
        share_text = f"{share:5.1f}%" if share is not None else "   N/A"
        print(f"{row['group']:<22}{row['graphrag_wins']:<10}"
              f"{row['vector_rag_wins']:<12}{row['ties']:<5}"
              f"{row['no_majority']:<13}{share_text}")
    print(f"\nSaved analysis to: {RESULTS_DIR}")
    print("The win rates describe judge preferences, not verified factual accuracy.")


if __name__ == "__main__":
    main()
