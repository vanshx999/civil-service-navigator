"""
Migrate existing ChromaDB documents into Elasticsearch.

Usage:
  python scripts/index_to_es.py

Requires ES_URL (or ELASTICSEARCH_URL) to be set in .env or env.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("index_to_es")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.agent import es_store
from src.agent.config import settings

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data", "chroma")
COLLECTION_NAME = "delhi_civic_sense"


def main():
    # 1. Connect to Chroma
    chroma_dir = os.path.abspath(CHROMA_DIR)
    if not os.path.exists(chroma_dir):
        logger.error(f"ChromaDB directory not found at {chroma_dir}")
        sys.exit(1)

    logger.info(f"Reading ChromaDB from {chroma_dir}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    # 2. Read all documents
    data = vectorstore.get()
    metadatas = data.get("metadatas", [])
    documents = data.get("documents", [])
    ids = data.get("ids", [])

    if not documents:
        logger.info("No documents found in ChromaDB")
        return

    logger.info(f"Read {len(documents)} documents from ChromaDB")

    # 3. Re-hydrate into LangChain Documents
    from langchain_core.documents import Document

    docs = []
    for i, (content, meta) in enumerate(zip(documents, metadatas)):
        docs.append(Document(page_content=content, metadata=meta or {}))

    # 4. Check ES connection
    es = es_store.get_es_store()
    if es is None:
        logger.error(
            "Could not connect to Elasticsearch. "
            "Set ES_URL (or ELASTICSEARCH_URL) in .env and ensure ES is running."
        )
        sys.exit(1)

    # 5. Index in batches
    indexed = es_store.index_documents(docs)
    logger.info(f"Successfully indexed {indexed}/{len(docs)} documents into ES index '{es_store.ES_INDEX}'")


if __name__ == "__main__":
    main()
