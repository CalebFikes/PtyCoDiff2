"""Create a complex-valued MNIST dataset by pairing random digits.

Each sample is formed as real = image_a / 255.0, imag = image_b / 255.0,
so the complex image is real + 1j * imag. The script saves train and test
sets as compressed npz files under `datasets/mnist/` with arrays:

- `images`: complex64 array shaped (N, H, W, 1)
- `labels_real`: int array shape (N,) digit labels for real part
- `labels_imag`: int array shape (N,) digit labels for imag part

The loader will try multiple backends (tensorflow, torchvision, sklearn).
"""

import os
import numpy as np
from typing import Tuple


def load_mnist_fallback() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Try tensorflow.keras, then torchvision, then sklearn.fetch_openml
    try:
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    try:
        import torchvision
        from torchvision.datasets import MNIST
        # torchvision returns PIL images; collect arrays
        train = MNIST(root='datasets/_raw', train=True, download=True)
        test = MNIST(root='datasets/_raw', train=False, download=True)
        x_train = np.stack([np.array(img) for img, _ in train], axis=0)
        y_train = np.array([label for _, label in train], dtype=np.int64)
        x_test = np.stack([np.array(img) for img, _ in test], axis=0)
        y_test = np.array([label for _, label in test], dtype=np.int64)
        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    try:
        from sklearn.datasets import fetch_openml

        mn = fetch_openml('mnist_784', version=1)
        X = mn['data'].astype(np.uint8)
        y = mn['target'].astype(np.int64)
        X = X.reshape(-1, 28, 28)
        x_train, y_train = X[:60000], y[:60000]
        x_test, y_test = X[60000:], y[60000:]
        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    # Final fallback: download raw MNIST IDX files from Yann LeCun's website
    try:
        import urllib.request
        import gzip

        def _read_idx(url):
            fn, _ = urllib.request.urlretrieve(url)
            with gzip.open(fn, 'rb') as f:
                data = f.read()
            return data

        base = 'http://yann.lecun.com/exdb/mnist'
        imgs_train = _read_idx(base + '/train-images-idx3-ubyte.gz')
        labels_train = _read_idx(base + '/train-labels-idx1-ubyte.gz')
        imgs_test = _read_idx(base + '/t10k-images-idx3-ubyte.gz')
        labels_test = _read_idx(base + '/t10k-labels-idx1-ubyte.gz')

        def parse_images(b):
            import struct
            magic, n, rows, cols = struct.unpack('>IIII', b[:16])
            arr = np.frombuffer(b, dtype=np.uint8, offset=16).reshape(n, rows, cols)
            return arr

        def parse_labels(b):
            import struct
            magic, n = struct.unpack('>II', b[:8])
            arr = np.frombuffer(b, dtype=np.uint8, offset=8).reshape(n,)
            return arr

        x_train = parse_images(imgs_train)
        y_train = parse_labels(labels_train)
        x_test = parse_images(imgs_test)
        y_test = parse_labels(labels_test)
        return x_train, y_train, x_test, y_test
    except Exception:
        pass

    # If network download fails or not available, fall back to sklearn 'digits' dataset
    try:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.images  # (n_samples, 8, 8)
        y = digits.target
        # upsample 8x8 -> 32x32 via nearest-neighbor then center-crop to 28x28
        X_up = np.repeat(np.repeat(X, 4, axis=1), 4, axis=2)
        n = X_up.shape[0]
        # center crop
        start = (32 - 28) // 2
        X_crop = X_up[:, start:start+28, start:start+28]
        # split into train/test (80/20)
        split = int(0.8 * n)
        x_train, y_train = X_crop[:split], y[:split]
        x_test, y_test = X_crop[split:], y[split:]
        return x_train.astype(np.uint8), y_train.astype(np.int64), x_test.astype(np.uint8), y_test.astype(np.int64)
    except Exception:
        raise ImportError(
            'No MNIST loader found and network fetch/load failed. Install tensorflow/torch/sklearn or enable network.'
        )


def make_complex_pairs(x: np.ndarray, y: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = x.shape[0]
    idx2 = rng.randint(0, N, size=N)

    real = x.astype(np.float32) / 255.0
    imag = x[idx2].astype(np.float32) / 255.0

    # shape to (N, H, W, 1)
    real = real[..., None]
    imag = imag[..., None]

    images = (real + 1j * imag).astype(np.complex64)
    labels_real = y.astype(np.int64)
    labels_imag = y[idx2].astype(np.int64)
    return images, labels_real, labels_imag


def save_npz(images: np.ndarray, labels_real: np.ndarray, labels_imag: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, images=images, labels_real=labels_real, labels_imag=labels_imag)


def main(outdir: str = 'datasets/mnist', seed: int = 0):
    x_train, y_train, x_test, y_test = load_mnist_fallback()

    train_images, train_lr, train_li = make_complex_pairs(x_train, y_train, seed=seed)
    test_images, test_lr, test_li = make_complex_pairs(x_test, y_test, seed=seed + 1)

    train_path = os.path.join(outdir, 'train.npz')
    test_path = os.path.join(outdir, 'test.npz')

    save_npz(train_images, train_lr, train_li, train_path)
    save_npz(test_images, test_lr, test_li, test_path)
    print(f'Saved train -> {train_path} ({train_images.shape}), test -> {test_path} ({test_images.shape})')


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='datasets/mnist')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(outdir=args.outdir, seed=args.seed)
