# evaluate.py

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BENCHMARK_DIR.parent

VECTOR_ANSWERS_FILE = BENCHMARK_DIR / "vector_answers.json"
GRAPHRAG_ANSWERS_FILE = BENCHMARK_DIR / "graphrag_answers.json"
OUTPUT_FILE = BENCHMARK_DIR / "evaluations.json"


# ---------------------------------------------------------
# ENVIRONMENT / JUDGE SETTINGS
# ---------------------------------------------------------

load_dotenv(PROJECT_DIR / ".env")

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
JUDGE_RUNS = int(os.getenv("JUDGE_RUNS", "3"))

client = OpenAI(api_key=os.getenv("GRAPHRAG_API_KEY"))


# ---------------------------------------------------------
# EVALUATION CRITERIA
# ---------------------------------------------------------

CRITERIA = {
    "comprehensiveness": (
        "How thoroughly does the answer cover the important aspects and "
        "details of the question? Prefer an answer that is complete and "
        "substantive without unnecessary repetition or irrelevant material."
    ),
    "diversity": (
        "How varied and multi-faceted is the answer? Prefer an answer that "
        "provides a richer range of relevant perspectives, concepts, "
        "connections, or insights rather than repeating one narrow point."
    ),
    "empowerment": (
        "How well does the answer help the reader understand the topic and "
        "make an informed judgment without being misled? Prefer clear "
        "explanations, reasoning, and useful supporting detail."
    ),
    "directness": (
        "How specifically and clearly does the answer address the question? "
        "Prefer a focused answer that answers what was asked without "
        "unnecessary material."
    ),
}


# ---------------------------------------------------------
# STRUCTURED OUTPUT SCHEMA
# ---------------------------------------------------------

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": (
                "1 if Answer 1 is better, 2 if Answer 2 is better, "
                "0 if the difference is immaterial or they are effectively tied."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "A short explanation of the decision using only the specified criterion."
            ),
        },
    },
    "required": ["winner", "reasoning"],
    "additionalProperties": False,
}


# ---------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def index_results(data):
    results = {}

    for item in data.get("results", []):
        question_id = item["question_id"]

        if question_id in results:
            raise ValueError(f"Duplicate question_id found: {question_id}")

        results[question_id] = item

    return results


# ---------------------------------------------------------
# MATCH VECTOR RAG AND GRAPHRAG ANSWERS
# ---------------------------------------------------------

def load_answer_pairs():
    vector_data = load_json(VECTOR_ANSWERS_FILE)
    graph_data = load_json(GRAPHRAG_ANSWERS_FILE)

    vector_results = index_results(vector_data)
    graph_results = index_results(graph_data)

    all_ids = sorted(set(vector_results) | set(graph_results))

    pairs = []

    for question_id in all_ids:
        vector_item = vector_results.get(question_id)
        graph_item = graph_results.get(question_id)

        if vector_item is None:
            print(f"Skipping {question_id}: missing Vector RAG answer.")
            continue

        if graph_item is None:
            print(f"Skipping {question_id}: missing GraphRAG answer.")
            continue

        if vector_item["question"] != graph_item["question"]:
            raise ValueError(f"Question text does not match for {question_id}.")

        pairs.append(
            {
                "question_id": question_id,
                "category": vector_item["category"],
                "question": vector_item["question"],
                "vector_answer": vector_item.get("answer"),
                "graphrag_answer": graph_item.get("answer"),
                "vector_error": vector_item.get("error"),
                "graphrag_error": graph_item.get("error"),
            }
        )

    return pairs


# ---------------------------------------------------------
# ANSWER ORDER RANDOMIZATION
# ---------------------------------------------------------

def should_swap(question_id, criterion, run_number):
    """
    Deterministically alternate which system is Answer 1 and Answer 2.
    This reduces position bias while keeping reruns reproducible.
    """
    value = f"{question_id}|{criterion}".encode("utf-8")
    digest = hashlib.sha256(value).digest()
    base_swap = digest[0] % 2

    return bool((base_swap + run_number - 1) % 2)


def arrange_answers(pair, criterion, run_number):
    swap = should_swap(pair["question_id"], criterion, run_number)

    if swap:
        return {
            "answer_1": pair["graphrag_answer"],
            "answer_1_system": "graphrag",
            "answer_2": pair["vector_answer"],
            "answer_2_system": "vector_rag",
        }

    return {
        "answer_1": pair["vector_answer"],
        "answer_1_system": "vector_rag",
        "answer_2": pair["graphrag_answer"],
        "answer_2_system": "graphrag",
    }


# ---------------------------------------------------------
# BUILD JUDGE PROMPT
# ---------------------------------------------------------

def build_judge_prompt(question, answer_1, answer_2, criterion):
    criterion_description = CRITERIA[criterion]

    return f"""
You are an impartial evaluator comparing two answers to the same question.

Judge the answers ONLY according to the evaluation criterion below.
Do not favor an answer because it is longer.
Do not assume either answer came from a particular retrieval system.
Do not use answer position as evidence of quality.

QUESTION:
{question}

EVALUATION CRITERION: {criterion.upper()}

{criterion_description}

ANSWER 1:
{answer_1}

ANSWER 2:
{answer_2}

Choose:
- 1 if Answer 1 is meaningfully better on this criterion.
- 2 if Answer 2 is meaningfully better on this criterion.
- 0 if they are effectively tied or the difference is immaterial.

Give a short explanation focused only on this criterion.
""".strip()


# ---------------------------------------------------------
# CALL THE LLM JUDGE
# ---------------------------------------------------------

def judge_answers(question, answer_1, answer_2, criterion):
    prompt = build_judge_prompt(
        question,
        answer_1,
        answer_2,
        criterion,
    )

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a neutral evaluator of retrieval-augmented "
                    "generation systems. Compare only the supplied answers "
                    "using the supplied criterion."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_judgment",
                "strict": True,
                "schema": JUDGMENT_SCHEMA,
            }
        },
    )

    judgment = json.loads(response.output_text)

    usage = getattr(response, "usage", None)

    token_usage = {
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }

    return judgment, token_usage


# ---------------------------------------------------------
# TRANSLATE ANSWER NUMBER INTO SYSTEM NAME
# ---------------------------------------------------------

def get_winner_system(judge_winner, ordering):
    if judge_winner == 0:
        return "tie"

    if judge_winner == 1:
        return ordering["answer_1_system"]

    if judge_winner == 2:
        return ordering["answer_2_system"]

    raise ValueError(f"Unexpected judge winner value: {judge_winner}")


# ---------------------------------------------------------
# SAVE / RESUME
# ---------------------------------------------------------

def save_evaluations(evaluations):
    output = {
        "benchmark_name": "Rousseau_RAG_vs_GraphRAG_Benchmark",
        "evaluation_method": "pairwise_llm_as_a_judge",
        "judge_model": JUDGE_MODEL,
        "judge_runs_per_criterion": JUDGE_RUNS,
        "criteria": list(CRITERIA.keys()),
        "total_evaluations": len(evaluations),
        "evaluations": evaluations,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


def load_existing_evaluations():
    if not OUTPUT_FILE.exists():
        return []

    data = load_json(OUTPUT_FILE)
    return data.get("evaluations", [])


def completed_keys(evaluations):
    """
    Completed or intentionally skipped evaluations do not need to run again.
    API errors are NOT marked finished, so rerunning evaluate.py will retry them.
    """
    return {
        (
            item["question_id"],
            item["criterion"],
            item["run_number"],
        )
        for item in evaluations
        if item.get("status") in {"completed", "skipped"}
    }


# ---------------------------------------------------------
# MAIN EVALUATION LOOP
# ---------------------------------------------------------

def run_evaluation():
    pairs = load_answer_pairs()

    evaluations = load_existing_evaluations()
    finished = completed_keys(evaluations)

    print("=" * 65)
    print("ROUSSEAU VECTOR RAG vs GRAPHRAG")
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 65)

    print(f"\nJudge model: {JUDGE_MODEL}")
    print(f"Questions available: {len(pairs)}")
    print(f"Criteria: {len(CRITERIA)}")
    print(f"Judge runs per criterion: {JUDGE_RUNS}")

    planned = len(pairs) * len(CRITERIA) * JUDGE_RUNS

    print(f"Maximum judge calls: {planned}")
    print(f"Already recorded: {len(evaluations)}\n")

    for question_number, pair in enumerate(pairs, start=1):
        question_id = pair["question_id"]

        generation_problem = None

        if pair["vector_error"]:
            generation_problem = (
                f"Vector RAG generation error: {pair['vector_error']}"
            )
        elif pair["graphrag_error"]:
            generation_problem = (
                f"GraphRAG generation error: {pair['graphrag_error']}"
            )
        elif not pair["vector_answer"]:
            generation_problem = "Vector RAG answer is empty."
        elif not pair["graphrag_answer"]:
            generation_problem = "GraphRAG answer is empty."

        print("-" * 65)
        print(
            f"Question {question_number}/{len(pairs)} "
            f"({question_id})"
        )
        print(f"Category: {pair['category']}")
        print(pair["question"])

        for criterion in CRITERIA:
            for run_number in range(1, JUDGE_RUNS + 1):
                key = (question_id, criterion, run_number)

                if key in finished:
                    print(
                        f"  {criterion} run {run_number}: already completed"
                    )
                    continue

                ordering = arrange_answers(
                    pair,
                    criterion,
                    run_number,
                )

                if generation_problem:
                    result = {
                        "question_id": question_id,
                        "category": pair["category"],
                        "question": pair["question"],
                        "criterion": criterion,
                        "run_number": run_number,
                        "answer_1_system": ordering["answer_1_system"],
                        "answer_2_system": ordering["answer_2_system"],
                        "judge_winner": None,
                        "winner_system": None,
                        "reasoning": None,
                        "judge_model": JUDGE_MODEL,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "status": "skipped",
                        "error": generation_problem,
                    }

                    evaluations.append(result)
                    finished.add(key)
                    save_evaluations(evaluations)

                    print(
                        f"  {criterion} run {run_number}: SKIPPED"
                    )
                    continue

                try:
                    judgment, token_usage = judge_answers(
                        pair["question"],
                        ordering["answer_1"],
                        ordering["answer_2"],
                        criterion,
                    )

                    judge_winner = judgment["winner"]
                    winner_system = get_winner_system(
                        judge_winner,
                        ordering,
                    )

                    result = {
                        "question_id": question_id,
                        "category": pair["category"],
                        "question": pair["question"],
                        "criterion": criterion,
                        "run_number": run_number,
                        "answer_1_system": ordering["answer_1_system"],
                        "answer_2_system": ordering["answer_2_system"],
                        "judge_winner": judge_winner,
                        "winner_system": winner_system,
                        "reasoning": judgment["reasoning"],
                        "judge_model": JUDGE_MODEL,
                        "input_tokens": token_usage["input_tokens"],
                        "output_tokens": token_usage["output_tokens"],
                        "total_tokens": token_usage["total_tokens"],
                        "status": "completed",
                        "error": None,
                    }

                    evaluations.append(result)
                    finished.add(key)
                    save_evaluations(evaluations)

                    print(
                        f"  {criterion} run {run_number}: {winner_system}"
                    )

                except Exception as error:
                    result = {
                        "question_id": question_id,
                        "category": pair["category"],
                        "question": pair["question"],
                        "criterion": criterion,
                        "run_number": run_number,
                        "answer_1_system": ordering["answer_1_system"],
                        "answer_2_system": ordering["answer_2_system"],
                        "judge_winner": None,
                        "winner_system": None,
                        "reasoning": None,
                        "judge_model": JUDGE_MODEL,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "status": "error",
                        "error": str(error),
                    }

                    evaluations.append(result)
                    finished.add(key)
                    save_evaluations(evaluations)

                    print(
                        f"  {criterion} run {run_number}: ERROR - {error}"
                    )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    print(f"\nResults saved to:\n{OUTPUT_FILE}")


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
