import cv2, numpy as np, random, re, subprocess, gzip, struct, sys, time, os
from PIL import Image, ImageDraw, ImageFont
from common import *

def load_idx(path):
    with gzip.open(path, 'rb') as f:
        d = f.read()
    if d[2] == 8 and d[3] == 3:
        n, r, c = struct.unpack('>III', d[4:16]); return np.frombuffer(d, np.uint8, offset=16).reshape(n, r, c)
    n = struct.unpack('>I', d[4:8])[0]; return np.frombuffer(d, np.uint8, offset=8)

# ---------- fonts
out = subprocess.run(['fc-list', ':', 'file'], capture_output=True, text=True).stdout.split('\n')
files = sorted({l.split(':')[0].strip() for l in out if l.strip()})
bad = re.compile(r'(symbol|dingbat|emoji|math|bamboo|music|arab|hebrew|devanagari|bengali|tamil|thai|lao|khmer|myanmar|georgian|armenian|ethiopic|gujarati|gurmukhi|kannada|malayalam|oriya|sinhala|telugu|tibetan|cuneiform|egyptian|adlam|cjk|mongolian|syriac|thaana|javanese|balinese|runic|ogham|glagolitic|gothic|phoenician|lisu|tifinagh|vai|yi|cherokee|canadian|osage|sundanese|batak|lepcha|limbu|nko|olchiki|rejang|saurashtra|samaritan|tagalog|tai|new_tai|cham|coptic|deseret|shavian|osmanya|lycian|lydian|carian|linear|ugaritic|old|brahmi|kaithi|kharoshthi|mandaic|meetei|cypriot|buginese|bopomofo|hanunoo|buhid|tagbanwa|kayah|mro|nushu|nyiakeng|pau|sora|takri|tirhuta|warang|wancho|zanabazar|ahom|anatolian|avestan|bassa|bhaiksuki|chakma|duployan|elbasan|elymaic|grantha|hatran|khojki|khudawadi|mahajani|manichaean|marchen|masaram|medefaidrin|meroitic|modi|multani|nabataean|newa|ottoman|pahawh|palmyrene|psalter|sharada|siddham|sogdian|soyombo|tangut|inscriptional|imperial|mende|miao|hmong|indic|lohit|padauk|abyssinica|kacst|ocr-b|opensymbol|wenquanyi|ipa|takao|droid|nanum|noto(?!sans-|serif-|sansdisplay|sansmono))', re.I)
fonts = []
for f in files:
    if not re.search(r'\.(ttf|otf|ttc)$', f, re.I): continue
    b = os.path.basename(f)
    if bad.search(b) and not re.match(r'^Noto(Sans|Serif)(Display|Mono)?-(Regular|Bold|Medium|SemiBold|Light|Italic|BoldItalic)\.(ttf|otf)$', b): continue
    fonts.append(f)
# verify each font really draws distinct digits (skip fonts that give boxes / blanks)
good = []
for f in fonts:
    try:
        ft = ImageFont.truetype(f, 48)
        imgs = []
        for ch in '1234567':
            im = Image.new('L', (80, 80), 0); ImageDraw.Draw(im).text((40, 40), ch, font=ft, fill=255, anchor='mm'); imgs.append(np.array(im))
        if all(i.sum() > 500 for i in imgs) and len({i.tobytes() for i in imgs}) == 7: good.append(f)
    except Exception: pass
print('fonts usable:', len(good), 'of', len(fonts), flush=True)
rng0 = random.Random(7); rng0.shuffle(good)
nval = max(10, len(good) // 8)
val_fonts, train_fonts = good[:nval], good[nval:]

def render(font_path, ch, rng):
    size = rng.randint(30, 60)
    font = ImageFont.truetype(font_path, size)
    im = Image.new('L', (110, 110), 0)
    ImageDraw.Draw(im).text((55, 55), ch, font=font, fill=255, anchor='mm', stroke_width=rng.choice([0, 0, 0, 1]))
    a = np.array(im)
    M = cv2.getRotationMatrix2D((55, 55), rng.uniform(-9, 9), rng.uniform(0.85, 1.15)); M[0, 1] += rng.uniform(-0.15, 0.15)
    a = cv2.warpAffine(a, M, (110, 110))
    k = rng.choice([0, 0, 2, 3])
    if k:
        a = cv2.dilate(a, np.ones((k, k), np.uint8)) if rng.random() < 0.6 else cv2.erode(a, np.ones((2, 2), np.uint8))
    return finish(a, rng)

NP = np.random.RandomState(5)
def elastic(a, alpha, sigma=8):
    h, w = a.shape
    dx = cv2.GaussianBlur(NP.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(NP.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(a, x + dx, y + dy, cv2.INTER_LINEAR)

def mnist_aug(img, rng):
    a = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.7: a = elastic(a, rng.uniform(40, 150))
    M = cv2.getRotationMatrix2D((42, 42), rng.uniform(-8, 8), rng.uniform(0.9, 1.1)); M[0, 1] += rng.uniform(-0.12, 0.12)
    a = cv2.warpAffine(a, M, (84, 84))
    k = rng.choice([0, 0, 2, 4, 5])
    if k: a = cv2.dilate(a, np.ones((k, k), np.uint8)) if rng.random() < 0.7 else cv2.erode(a, np.ones((3, 3), np.uint8))
    return finish(a, rng)

def build(fontlist, per, seed):
    rng = random.Random(seed); X, y = [], []
    for f in fontlist:
        for d in range(1, 10):
            for _ in range(per):
                try: v = render(f, str(d), rng)
                except Exception: v = None
                if v is not None: X.append(v); y.append(d - 1)
    return np.array(X, np.float32), np.array(y, np.int8)

t = time.time()
Xf, yf = build(train_fonts, 26, 1); print('train fonts', Xf.shape, round(time.time() - t), 's', flush=True)
Xv, yv = build(val_fonts, 6, 2); print('val fonts', Xv.shape, flush=True)

base = '/tmp/mn/'
tri, trl = load_idx(base + 'train-images-idx3-ubyte.gz'), load_idx(base + 'train-labels-idx1-ubyte.gz')
tei, tel = load_idx(base + 't10k-images-idx3-ubyte.gz'), load_idx(base + 't10k-labels-idx1-ubyte.gz')
rng = random.Random(3)
Xm, ym = [], []
for copy in range(3):
    for img, l in zip(tri, trl):
        if l == 0: continue
        v = mnist_aug(img, rng)
        if v is not None: Xm.append(v); ym.append(l - 1)
    print('mnist copy', copy, flush=True)
Xm, ym = np.array(Xm, np.float32), np.array(ym, np.int8); print('mnist train', Xm.shape, round(time.time() - t), 's', flush=True)
Xmt, ymt = [], []
for img, l in zip(tei, tel):
    if l == 0: continue
    v = mnist_aug(img, rng)
    if v is not None: Xmt.append(v); ymt.append(l - 1)
Xmt, ymt = np.array(Xmt, np.float32), np.array(ymt, np.int8)
np.savez_compressed('data.npz', Xf=Xf, yf=yf, Xv=Xv, yv=yv, Xm=Xm, ym=ym, Xmt=Xmt, ymt=ymt)
print('saved', round(time.time() - t), 's', flush=True)
