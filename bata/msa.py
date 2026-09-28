"""ProteinGym full-length alignment slicing; preserves frozen row multiplicity."""
import numpy as np


def region_codes(path, full_length, start, stop, mode='aligned'):
    alphabet = '-ACDEFGHIKLMNPQRSTVWY'
    rows, chunks = [], []
    raw_count = 0

    def append_record():
        nonlocal raw_count
        raw_count += 1
        full = ''.join(chunks)
        if mode == 'jackhmmer_grouped':
            full = ''.join(c for c in full if not c.islower() and c != '.')
        else:
            full = full.replace('.', '-').upper()
        if len(full) != full_length:
            raise ValueError('Alignment record length mismatch')
        region = full[start - 1:stop]
        if region.count('-') / len(region) <= 0.5 and all(c in alphabet for c in region):
            rows.append([alphabet.index(c) for c in region])

    if mode not in ('aligned', 'jackhmmer_grouped'):
        raise ValueError('Unknown frozen MSA rule')
    if not 1 <= start <= stop <= full_length:
        raise ValueError('Invalid region coordinates')
    seen_header = False
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seen_header:
                    append_record()
                seen_header, chunks = True, []
            else:
                if not seen_header:
                    raise ValueError('Sequence before FASTA header')
                chunks.append(''.join(line.split()))
    if seen_header:
        append_record()
    if not rows:
        raise ValueError('No valid region sequences')
    if mode == 'jackhmmer_grouped':
        # Frozen fitter expands counts in first-occurrence order, not raw order.
        from collections import Counter
        counts = Counter(tuple(row) for row in rows)
        rows = [row for row, count in counts.items() for _ in range(count)]
    return np.asarray(rows, dtype=np.uint8), raw_count
