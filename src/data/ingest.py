import os
import glob
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_documents() -> list[Document]:
    docs = []
    md_files = glob.glob(os.path.join(RAW_DIR, "*.md"))
    if not md_files:
        print(f"ERROR: No markdown files found in {RAW_DIR}")
        return docs

    for filepath in sorted(md_files):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse metadata from the file header
        source_url = ""
        source_name = ""
        lines = content.split("\n")
        for line in lines[:10]:
            if line.startswith("- **Source URL**:"):
                source_url = line.split(":", 1)[1].strip()
            if line.startswith("- **Source**:"):
                source_name = line.split(":", 1)[1].strip()

        # Skip header section (before ---)
        body_start = content.find("---\n\n")
        if body_start != -1:
            body = content[body_start + 4:]
        else:
            body = content

        # Skip files whose body (after the header) is under 300 characters
        body = body.strip()
        if len(body) < 300:
            print(f"  WARNING: Skipping {os.path.basename(filepath)} — body too short ({len(body)} chars, min 300)")
            continue

        doc = Document(
            page_content=body,
            metadata={
                "source": source_name or filepath,
                "source_url": source_url,
                "file": os.path.basename(filepath),
            }
        )
        docs.append(doc)
        print(f"  Loaded: {os.path.basename(filepath)} ({len(body)} chars)")

    print(f"\nTotal documents loaded: {len(docs)}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = hashlib.md5(
            f"{chunk.metadata.get('source_url', '')}_{i}".encode()
        ).hexdigest()[:12]
        chunk.metadata["chunk_index"] = i

    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    print("\nInitializing embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Creating ChromaDB at {CHROMA_DIR}...")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="delhi_civic_sense",
    )

    print(f"Vectorstore created with {len(chunks)} chunks")
    return vectorstore


def main():
    print("=" * 60)
    print("Delhi Civic Sense Navigator - RAG Ingestion Pipeline")
    print("=" * 60)

    docs = load_documents()
    if not docs:
        print("No documents to process. Run scraper.py first.")
        return

    chunks = chunk_documents(docs)
    if not chunks:
        print("No chunks created.")
        return

    vectorstore = build_vectorstore(chunks)

    print(f"\n{'=' * 60}")
    print(f"Done! ChromaDB stored at: {CHROMA_DIR}")
    print(f"Collection: delhi_civic_sense")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding model: all-MiniLM-L6-v2")
    print("=" * 60)


if __name__ == "__main__":
    main()
