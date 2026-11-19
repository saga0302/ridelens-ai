"""Build the RAG index over the README and the glossary.

Run from the project root: python -m rag.build_index
"""
import os
import re

import chromadb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(ROOT, "rag", "chroma")
COLLECTION = "ridelens_docs"
SOURCES = ["README.md", os.path.join("docs", "glossary.md")]


def split_sections(path):
    """Split a markdown file into (heading, text) chunks at each '## ' heading."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"(?m)^## ", text)
    chunks = []
    intro = parts[0].strip()
    if intro:
        chunks.append((os.path.basename(path), intro))
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        chunks.append((heading.strip(), f"## {heading}\n{body}".strip()))
    return chunks


def build():
    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each time
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    ids, docs, metas = [], [], []
    for src in SOURCES:
        for i, (heading, text) in enumerate(split_sections(src)):
            ids.append(f"{src}#{i}")
            docs.append(text)
            metas.append({"source": src, "heading": heading})
    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"indexed {len(ids)} sections from {len(SOURCES)} files")
    return len(ids)


if __name__ == "__main__":
    build()