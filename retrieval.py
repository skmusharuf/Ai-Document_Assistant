from ingestion import get_vector_store
from config import DEFAULT_TOP_K


def retrieve(question: str, top_k: int = DEFAULT_TOP_K):
    store = get_vector_store()
    if store._collection.count() == 0:
        return []

    docs_with_scores = store.similarity_search_with_score(
        question, k=min(top_k, store._collection.count())
    )
    results = []
    for doc, distance in docs_with_scores:
        results.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "Unknown source"),
            "page": doc.metadata.get("page_number", int(doc.metadata.get("page", 0)) + 1),
            "distance": float(distance),
            "metadata": doc.metadata,
        })
    return results
