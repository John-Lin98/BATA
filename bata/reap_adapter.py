"""Frozen REAP wrapper; requires separately installed upstream reap package.

No upstream REAP implementation is redistributed here. Formal training uses
CUDA, 100 members and the original stopping rule, never unseen fitness labels.
"""
import numpy as np
from sklearn.model_selection import train_test_split

def reap_scores(data, train, y, keys, algorithm_seed, rnd, members=5):
    from reap.training import train_plm_rankreg, predict_plm_model
    (predictions, metadata) = ([], [])
    for member in range(members):
        seed = int(np.random.SeedSequence([algorithm_seed, rnd, 1701, member]).generate_state(1)[0])
        (ti, vi) = train_test_split(np.arange(len(train)), test_size=0.1, random_state=seed, shuffle=True)
        (yy, yv) = (y[ti], y[vi])
        (ym, ys) = (float(yy.mean()), float(yy.std()))
        ys = ys if ys >= 1e-12 else 1.0
        (rho, mse, _, model) = train_plm_rankreg(X_train=data.feature('plm', train[ti]), y_train=(yy - ym) / ys, X_val=data.feature('plm', train[vi]), y_val=(yv - ym) / ys, epochs=1000, seed=seed, save_path=None, model_type='mlp', alpha=0.8, margin=0.001, patience=50, batch_size=1024, lr=0.0003, wd=1e-05, device='cuda')
        p = np.empty(len(keys), dtype=np.float64)
        for start in range(0, len(keys), 8192):
            kk = keys[start:start + 8192]
            p[start:start + len(kk)] = predict_plm_model(model, data.feature('plm', kk), batch_size=4096, device='cuda') * ys + ym
        assert np.isfinite(p).all()
        predictions.append(p)
        metadata.append(dict(member=member, seed=seed, train_ids=train[ti].tolist(), val_ids=train[vi].tolist(), mean=ym, scale=ys, rho=float(rho), mse=float(mse)))
        del model
    return (np.stack(predictions, axis=1), metadata)

def reap_select(data, train, y, keys, algorithm_seed, rnd):
    from reap.selection import rank_by_ucb
    scores, metadata = reap_scores(data, train, y, keys, algorithm_seed, rnd, members=100)
    weight = .5 if rnd <= 3 else 0.
    order, _ = rank_by_ucb(scores.mean(axis=1), scores.std(axis=1), lambda_sigma=weight, top_n=96)
    return [(int(j), -1) for j in order], scores, dict(fine_head='reap100', members=metadata, lambda_sigma=weight)
