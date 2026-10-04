# run_graphrag.py

import json
import time
import subprocess
from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BENCHMARK_DIR.parent


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

QUESTIONS_FILE = BENCHMARK_DIR / "questions.json"

OUTPUT_FILE = BENCHMARK_DIR / "graphrag_answers.json"


# ---------------------------------------------------------
# GRAPHRAG SETTINGS
# ---------------------------------------------------------

QUERY_METHOD = "global"


# ---------------------------------------------------------
# LOAD QUESTIONS
# ---------------------------------------------------------

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# ---------------------------------------------------------
# ASK GRAPHRAG
# ---------------------------------------------------------

def ask_graphrag(question):
    """
    Sends a question to the existing GraphRAG index
    using the GraphRAG command line interface.
    """

    command = [
        "graphrag",
        "query",
        question,
        "--root",
        str(PROJECT_DIR),
        "--method",
        QUERY_METHOD
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # If GraphRAG failed, show the actual error
    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.strip()
        )

    return result.stdout.strip()


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(benchmark_name, results):

    output = {
        "benchmark_name": benchmark_name,
        "system": "graphrag",
        "query_method": QUERY_METHOD,
        "total_completed": len(results),
        "results": results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# RUN BENCHMARK
# ---------------------------------------------------------

def run_benchmark():

    benchmark = load_questions()

    questions = benchmark["questions"]

    results = []

    print("=" * 60)
    print("ROUSSEAU GRAPHRAG BENCHMARK")
    print("=" * 60)

    print(
        f"\nGraphRAG query method: {QUERY_METHOD}"
    )

    print(
        f"Total questions: {len(questions)}\n"
    )


    for number, item in enumerate(
        questions,
        start=1
    ):

        question_id = item["id"]

        category = item["category"]

        question = item["question"]


        print("-" * 60)

        print(
            f"Question {number}/{len(questions)}"
        )

        print(
            f"ID: {question_id}"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Question: {question}"
        )


        # Start timer
        start_time = time.time()


        try:

            answer = ask_graphrag(
                question
            )

            error = None


        except Exception as e:

            answer = None

            error = str(e)

            print(
                f"ERROR: {error}"
            )


        # Stop timer
        end_time = time.time()

        latency = end_time - start_time


        print(
            f"Response time: "
            f"{latency:.2f} seconds"
        )


        result = {

            "question_id":
                question_id,

            "category":
                category,

            "difficulty":
                item["difficulty"],

            "question":
                question,

            "expected_scope":
                item["expected_scope"],

            "target_documents":
                item["target_documents"],

            "requires_cross_document_reasoning":
                item[
                    "requires_cross_document_reasoning"
                ],

            "requires_global_sensemaking":
                item[
                    "requires_global_sensemaking"
                ],

            "system":
                "graphrag",

            "query_method":
                QUERY_METHOD,

            "answer":
                answer,

            "latency_seconds":
                round(
                    latency,
                    3
                ),

            "error":
                error
        }


        results.append(
            result
        )


        # Save after every question
        save_results(
            benchmark["benchmark_name"],
            results
        )


    print(
        "\n" + "=" * 60
    )

    print(
        "BENCHMARK COMPLETE"
    )

    print(
        "=" * 60
    )


    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

if __name__ == "__main__":

    run_benchmark()