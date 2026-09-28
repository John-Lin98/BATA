"""Frozen 96 + 4 x 96 evaluation; selector receives revealed feedback only."""
import math


def validate_batch(indices, seen, size):
    if (len(indices) != 96 or any(type(i) is not int for i in indices)
            or len(set(indices)) != 96 or set(indices).intersection(seen)
            or any(i < 0 or i >= size for i in indices)):
        raise ValueError('Expected 96 distinct, unseen, in-range integer indices')


def closed_loop(initial, size, select, reveal):
    observed, history = {}, []
    batch = list(initial)
    for round_number in range(5):
        if round_number:
            batch = list(select(round_number, dict(sorted(observed.items()))))
        validate_batch(batch, observed, size)
        # All 96 identities are fixed and validated before requesting any label.
        values = reveal(tuple(batch))
        if set(values) != set(batch) or not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Incomplete or invalid batch feedback')
        observed.update(values)
        history.append(float(max(observed.values())))
    return dict(queries=len(observed), unique_queries=len(observed), history=history,
                Final=history[-1], Query_AUC=(history[0] + 2 * sum(history[1:4]) + history[4]) / 8)
