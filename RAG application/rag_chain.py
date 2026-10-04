# rag_chain.py

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from database import get_vector_store

load_dotenv()


PROMPT_TEMPLATE = """
You are an academic assistant specializing in Spinoza's Ethics.

Use the provided excerpts as your philosophical doctrine and basis of interpretation.

Your task is not only to quote the excerpts, but also to reason from them.
You may apply Spinoza's ideas to hypothetical scenarios, moral questions, political situations, and human behavior.

When answering:
- Explain the relevant Spinozist concepts from the excerpts.
- Apply those concepts carefully to the user's scenario.
- Make clear when you are interpreting beyond the literal wording.
- Do not invent fake citations or pretend the excerpts say something they do not say.
- If the excerpts are too limited, give the best partial interpretation and say what is missing.

Context excerpts:
{context}

User question:
{question}

Answer:
"""


def ask_rag(question):
    """
    You are an academic assistant specializing in Spinoza philosophy.

    Answer the user's question using the article excerpts below as a doctrine.

    You are to philosophize in hypothetical scenarios using the article excerpts.
    
    If the excerpts do not contain enough information, say:
    "The provided articles do not contain enough information to answer that accurately."

    """

    # 1. Connect to Chroma
    vector_store = get_vector_store()

    # 2. Search for relevant chunks
    results = vector_store.similarity_search_with_score(
        question,
        k=5
    )

    # 3. If no results are found
    if len(results) == 0:
        return "I could not find anything relevant in the database."

    # 4. Combine retrieved chunks into one context string
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, score in results]
    )

    # 5. Build the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(
        context=context_text,
        question=question
    )

    # 6. Create the LLM
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 7. Ask the LLM
    response = model.invoke(prompt)

    return response.content