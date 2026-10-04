# database.py

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "spinoza_collection"


def get_embedding_function():
    """
    Creates the embedding model.
    This turns text into vectors.
    """
    return OpenAIEmbeddings(model="text-embedding-3-large")


def get_vector_store():
    """
    Opens the Chroma vector database.
    If chroma_db/ does not exist yet, Chroma can create it.
    """
    embedding_function = get_embedding_function()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    return vector_store