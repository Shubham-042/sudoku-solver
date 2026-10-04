import numpy as np, time, struct, warnings
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings('ignore')
d = np.load('data.npz')
X = np.vstack([d['Xf'], d['Xm']]); y = np.concatenate([d['yf'], d['ym']]).astype(int)
print('train', X.shape, flush=True)
Xv, yv, Xmt, ymt = d['Xv'], d['yv'], d['Xmt'], d['ymt']
models = []
for seed, hidden in [(1, (384, 192)), (2, (384, 192)), (3, (256, 128))]:
    idx = np.random.RandomState(seed).permutation(len(X))
    Xs, ys = X[idx], y[idx]
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation='relu', solver='adam', batch_size=256,
                        learning_rate_init=1e-3, alpha=3e-4, max_iter=1, warm_start=True, random_state=seed)
    t = time.time()
    for ep in range(1, 17):
        if ep == 11: clf.learning_rate_init = 3e-4
        if ep == 15: clf.learning_rate_init = 1e-4
        clf.fit(Xs, ys)
    fv = (clf.predict(Xv) == yv).mean(); mv = (clf.predict(Xmt) == ymt).mean()
    print(f'model {seed} {hidden}: unseen-fonts {fv:.4f}  mnist-test {mv:.4f}  {time.time()-t:.0f}s', flush=True)
    models.append(clf)
P = lambda Xe: sum(m.predict_proba(Xe) for m in models) / len(models)
print(f'ENSEMBLE: unseen-fonts {(P(Xv).argmax(1)==yv).mean():.4f}  mnist-test {(P(Xmt).argmax(1)==ymt).mean():.4f}', flush=True)
with open('digits.bin', 'wb') as f:
    f.write(struct.pack('>i', len(models)))
    for clf in models:
        f.write(struct.pack('>i', len(clf.coefs_)))
        for W, b in zip(clf.coefs_, clf.intercepts_):
            f.write(struct.pack('>ii', W.shape[0], W.shape[1]))
            f.write(W.astype('>f4').tobytes()); f.write(b.astype('>f4').tobytes())
np.save('ref_x.npy', Xv[:40]); np.save('ref_probs.npy', P(Xv[:40]))
print('exported', flush=True)
