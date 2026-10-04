# How digits.bin was made (optional - already included in src/main/resources)

Python 3 with numpy, opencv-python, pillow, scikit-learn. MNIST idx files in /tmp/mn/ (see generate_data.py).

    python generate_data.py        # MNIST (handwritten, elastic-distorted) + digits rendered from ~340 fonts
    python train_and_export.py     # trains 3 small networks, writes digits.bin (copy it to src/main/resources/)

Accuracy on held-out data: 99.2% on fonts never seen in training, ~97% on distorted MNIST test digits.
