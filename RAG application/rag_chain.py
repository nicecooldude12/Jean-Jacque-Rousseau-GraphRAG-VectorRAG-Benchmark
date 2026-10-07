# rag_chain.py

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from database import get_vector_store

load_dotenv()


PROMPT_TEMPLATE = """
You are an academic assistant answering questions about
Jean-Jacques Rousseau's writings.

Base your answer on the provided excerpts.

When answering:
- Address the user's question directly.
- Explain relevant concepts clearly.
- Connect ideas across the excerpts when the evidence supports it.
- Distinguish statements supported by the excerpts from your interpretations.
- Do not introduce unsupported claims or invent quotations or citations.
- If the excerpts support only part of the question, answer that part
  and clearly explain what information is missing.
- If the excerpts do not support an answer, say:
  "The provided excerpts do not contain enough information to answer
  this question."
- Include relevant detail while avoiding repetition and unrelated material.

Context excerpts:
{context}

User question:
{question}

Answer:
"""


def ask_rag(question):
    """
    Retrieve passages from the indexed corpus and generate
    an evidence-based answer about Rousseau.
    """

    # 1. Connect to the existing Chroma database.
    vector_store = get_vector_store()

    # 2. Retrieve up to five passages.
    results = vector_store.similarity_search_with_score(
        question,
        k=5
    )

    # 3. Handle an empty retrieval result.
    if not results:
        return (
            "The provided excerpts do not contain enough information "
            "to answer this question."
        )

    # 4. Combine the retrieved passages into the model's context.
    context_text = "\n\n---\n\n".join(
        doc.page_content for doc, score in results
    )

    # 5. Build the answer-generation prompt.
    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.format(
        context=context_text,
        question=question
    )

    # 6. Use the same answer-generation model configured for GraphRAG.
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    # 7. Generate and return the answer.
    response = model.invoke(prompt)

    return response.content
