"""Find the passages most relevant to a question."""
import chromadb

from rag.build_index import COLLECTION, INDEX_DIR


def search(question, n=3):
    client = chromadb.PersistentClient(path=INDEX_DIR)
    collection = client.get_collection(COLLECTION)
    res = collection.query(query_texts=[question], n_results=n)
    return [
        {"source": meta["source"], "heading": meta["heading"], "text": doc, "distance": dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]