import json
import time
from pathlib import Path
import sys

BENCHMARK_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BENCHMARK_DIR.parent

RAG_APP_DIR = PROJECT_DIR / "RAG application"


# Let Python find rag_chain.py, database.py, etc.
sys.path.insert(0, str(RAG_APP_DIR))


# Import your existing Vector RAG function
from rag_chain import ask_rag


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

QUESTIONS_FILE = BENCHMARK_DIR / "questions.json"

OUTPUT_FILE = BENCHMARK_DIR / "vector_answers.json"


# ---------------------------------------------------------
# LOAD QUESTIONS
# ---------------------------------------------------------

def load_questions():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_results(benchmark_name, results):

    output = {
        "benchmark_name": benchmark_name,
        "system": "vector_rag",
        "total_completed": len(results),
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

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
    print("ROUSSEAU VECTOR RAG BENCHMARK")
    print("=" * 60)

    print(f"\nTotal questions: {len(questions)}\n")


    for number, item in enumerate(questions, start=1):

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

            # This calls your existing RAG system
            answer = ask_rag(question)

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
            f"Response time: {latency:.2f} seconds"
        )


        # Save this result
        result = {

            "question_id": question_id,

            "category": category,

            "difficulty": item["difficulty"],

            "question": question,

            "expected_scope":
                item["expected_scope"],

            "target_documents":
                item["target_documents"],

            "requires_cross_document_reasoning":
                item["requires_cross_document_reasoning"],

            "requires_global_sensemaking":
                item["requires_global_sensemaking"],

            "system": "vector_rag",

            "answer": answer,

            "latency_seconds":
                round(latency, 3),

            "error": error
        }


        results.append(result)


        # Save after every question
        save_results(
            benchmark["benchmark_name"],
            results
        )


    print("\n" + "=" * 60)

    print("BENCHMARK COMPLETE")

    print("=" * 60)


    print(
        f"\nResults saved to:\n{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

if __name__ == "__main__":

    run_benchmark()