# app.py

from rag_chain import ask_rag


def main():
    print("RAG chatbot is ready.")
    print("Ask a question about your stored documents.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        answer = ask_rag(question)

        print("\nAnswer:")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()