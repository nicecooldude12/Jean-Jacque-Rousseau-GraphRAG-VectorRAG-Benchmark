# ingest.py

import os
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database import get_vector_store


CHROMA_PATH = "chroma_db"
DATA_FOLDER = "RAG application\\data"


def load_text_files(folder_path):
    """
    Loads every .txt file inside the data folder.
    """

    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Could not find folder: {folder_path}"
        )

    documents = []

    # Loop through every file in the folder
    for filename in os.listdir(folder_path):

        # Only use .txt files
        if filename.endswith(".txt"):

            file_path = os.path.join(
                folder_path,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            if not text.strip():
                print(
                    f"Skipping empty file: {filename}"
                )
                continue

            print(
                f"Loaded: {filename}"
            )

            print(
                f"Document length: {len(text)}"
            )

            document = Document(
                page_content=text,

                metadata={
                    "source": filename
                }
            )

            documents.append(document)

    if len(documents) == 0:
        raise ValueError(
            "No .txt documents were found."
        )

    print()
    print(
        f"Total documents loaded: {len(documents)}"
    )

    return documents


def split_documents(documents):
    """
    Splits all documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    if len(chunks) == 0:
        raise ValueError(
            "No chunks were created."
        )

    return chunks


def save_to_chroma(chunks):
    """
    Stores all document chunks in Chroma.
    """

    if len(chunks) == 0:
        raise ValueError(
            "Cannot save empty chunks."
        )

    vector_store = get_vector_store()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    print(
        f"Number of IDs created: {len(ids)}"
    )

    print(
        "First chunk source:",
        chunks[0].metadata["source"]
    )

    print(
        "First chunk preview:",
        chunks[0].page_content[:100]
    )

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    print(
        f"Saved {len(chunks)} chunks "
        "to Chroma database."
    )


def main():

    # Reset database before rebuilding
    if os.path.exists(CHROMA_PATH):

        shutil.rmtree(CHROMA_PATH)

        print(
            "Old Chroma database deleted."
        )

    documents = load_text_files(
        DATA_FOLDER
    )

    chunks = split_documents(
        documents
    )

    save_to_chroma(
        chunks
    )


if __name__ == "__main__":
    main()