import json
from pathlib import Path

from retrieval import retrieve

TEST_FILE = Path(__file__).resolve().parent / "evaluation_questions.json"
K = 4


def main():
    cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    if not cases:
        print("No evaluation questions found.")
        return

    hits = 0
    reciprocal_rank_sum = 0.0
    precision_sum = 0.0

    for case in cases:
        question = case["question"]
        expected = {s.casefold() for s in case["expected_sources"]}
        results = retrieve(question, top_k=K)
        retrieved = [r["source"].casefold() for r in results]

        relevant_ranks = [
            rank for rank, source in enumerate(retrieved, start=1)
            if source in expected
        ]
        hit = bool(relevant_ranks)
        hits += int(hit)
        if hit:
            reciprocal_rank_sum += 1.0 / relevant_ranks[0]

        # Precision@K uses K as the denominator, even if fewer than K results exist.
        precision_sum += sum(source in expected for source in retrieved) / K

        print(f"\nQuestion: {question}")
        print(f"Expected source files: {sorted(expected)}")
        print(f"Retrieved source files: {retrieved}")
        print(f"Hit@{K}: {int(hit)}")

    n = len(cases)
    hit_rate = hits / n
    mrr = reciprocal_rank_sum / n
    precision_at_k = precision_sum / n

    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Questions:       {n}")
    print(f"Hit Rate@{K}:    {hit_rate:.3f}")
    print(f"Recall@{K}*:     {hit_rate:.3f}")
    print(f"Precision@{K}:   {precision_at_k:.3f}")
    print(f"MRR:             {mrr:.3f}")
    print("==========================================")
    print("*Recall@K is a source-level proxy here, not chunk-level recall.")
    print("These scores are meaningful only if expected_sources are correctly labeled.")


if __name__ == "__main__":
    main()
