def compute_repetition_score(data: bytes, n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n must be greater than 0")

    total_ngrams = len(data) - n + 1
    if total_ngrams <= 0:
        return 0.0

    unique_ngrams = {data[i : i + n] for i in range(total_ngrams)}
    ratio = len(unique_ngrams) / total_ngrams
    return float(min(1.0, max(0.0, ratio)))
