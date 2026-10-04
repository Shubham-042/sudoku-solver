import cv2, numpy as np, math
def norm_patch(patch):
    """patch: uint8 tight crop, ink = white. Returns float32[784] (MNIST-style 28x28, digit fitted in 20px, centred by mass)."""
    h, w = patch.shape
    s = 20.0 / max(h, w)
    nw, nh = max(1, int(math.floor(w * s + 0.5))), max(1, int(math.floor(h * s + 0.5)))
    r = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((28, 28), np.uint8)
    x0, y0 = (28 - nw) // 2, (28 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    m = cv2.moments(canvas)
    if m['m00'] > 0:
        cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))
    return canvas.astype(np.float32).ravel() / 255.0

def crop_ink(b):
    ys, xs = np.where(b > 0)
    if len(ys) < 15: return None
    return b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def clip_aug(b, rng):
    """simulate grid-line removal / lines touching the digit: shave a few pixels off one side"""
    if rng.random() < 0.18:
        k = rng.randint(1, 3); side = rng.randint(0, 3)
        if side == 0: b[:k, :] = 0
        elif side == 1: b[-k:, :] = 0
        elif side == 2: b[:, :k] = 0
        else: b[:, -k:] = 0
    return b

def finish(a, rng):
    blur = rng.uniform(0, 1.4)
    if blur > 0.3: a = cv2.GaussianBlur(a, (0, 0), blur)
    thr = rng.randint(70, 170)
    b = (a > thr).astype(np.uint8) * 255
    b = clip_aug(b, rng)
    c = crop_ink(b)
    return None if c is None else norm_patch(c)
