def compute_rle_ratio(data: bytes) -> float:
    if not data:
        return 0.0

    runs = 1
    prev = data[0]

    for byte in data[1:]:
        if byte != prev:
            runs += 1
            prev = byte

    ratio = runs / len(data)
    return float(min(1.0, max(0.0, ratio)))
