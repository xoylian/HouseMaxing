"""
ingest.py — One-time script to embed the knowledge base into ChromaDB.
Run this ONCE before launching the Streamlit app:
    python ingest.py
"""

import json
import os
import sys
from pathlib import Path
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# ─── Config ────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_PATH = "knowledge_base_airbnb.json"
CHROMA_PERSIST_DIR  = "./chroma_db"
COLLECTION_NAME     = "house_check_rules"
EMBEDDING_MODEL     = "text-embedding-3-small"
# ───────────────────────────────────────────────────────────────────────────────


def load_rules(path: str) -> List[dict]:
    """Load and validate the knowledge base JSON."""
    kb_path = Path(path)
    if not kb_path.exists():
        print(f"[ERROR] knowledge_base.json not found at: {kb_path.resolve()}")
        sys.exit(1)

    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both flat array format and {"rules": [...]} wrapper format
    rules = data if isinstance(data, list) else data.get("rules", [])
    if not rules:
        print("[ERROR] No rules found in knowledge base file.")
        sys.exit(1)

    print(f"[✓] Loaded {len(rules)} rules from knowledge base.")
    return rules


def rules_to_documents(rules: List[dict]) -> List[Document]:
    """
    Convert each rule into a LangChain Document.
    The page_content is a rich semantic string for embedding;
    all structured fields live in metadata for retrieval.
    """
    docs = []
    for rule in rules:
        # Normalize: support both "issue" (airbnb file) and "title" (original file)
        title = rule.get("issue") or rule.get("title", "")

        # Rich text blob that captures the full semantic meaning for embeddings
        content = (
            f"Category: {rule['category']}. "
            f"Issue: {title}. "
            f"Description: {rule['description']} "
            f"Recommended Action: {rule['action']}"
        )
        metadata = {
            "id":          rule["id"],
            "category":    rule["category"],
            "title":       title,
            "severity":    rule["severity"],
            "action":      rule["action"],
            "description": rule["description"],
            "photo_tip":   rule.get("photo_tip", ""),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    print(f"[✓] Converted {len(docs)} rules to LangChain Documents.")
    return docs


def ingest(docs: list[Document]) -> Chroma:
    """Embed and persist all documents into ChromaDB."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("\n🔑 Enter your OpenAI API key (starts with sk-):")
        api_key = input("   API Key: ").strip()
        if not api_key.startswith("sk-"):
            print("[ERROR] That doesn't look like a valid key. It should start with sk-")
            sys.exit(1)
        print("[✓] API key accepted.\n")

    print(f"[…] Embedding {len(docs)} documents using '{EMBEDDING_MODEL}'…")
    print("     (This calls the OpenAI API once — subsequent app runs load from disk)")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    # Wipe and recreate the collection for a clean ingest
    persist_path = Path(CHROMA_PERSIST_DIR)
    if persist_path.exists():
        import shutil
        shutil.rmtree(persist_path)
        print(f"[✓] Cleared existing ChromaDB at {CHROMA_PERSIST_DIR}")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    count = vectorstore._collection.count()
    print(f"[✓] Successfully ingested {count} vectors into ChromaDB.")
    print(f"[✓] Persisted to: {persist_path.resolve()}")
    return vectorstore


def verify(vectorstore: Chroma):
    """Run a quick sanity-check query to confirm retrieval works."""
    print("\n[…] Running verification query: 'bedroom dark messy no pillows'")
    results = vectorstore.similarity_search("bedroom dark messy no pillows", k=3)
    print(f"[✓] Top {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"    {i}. [{doc.metadata['severity']}] {doc.metadata['id']} — {doc.metadata['title']}")
    print("\n[🚀] Ingestion complete. You can now run: streamlit run app.py\n")


if __name__ == "__main__":
    rules    = load_rules(KNOWLEDGE_BASE_PATH)
    docs     = rules_to_documents(rules)
    vs       = ingest(docs)
    verify(vs)