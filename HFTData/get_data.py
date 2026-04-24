#!/usr/bin/env python3
"""
Download all classic ML and image datasets needed for experiments.
Saves headerless CSVs (features + classes) to ../data/.

Datasets:
  - iris        (150 x 4)
  - wine        (178 x 13)
  - cancer      (569 x 30)
  - digits      (1797 x 64)
  - mnist       (70000 x 784, capped to 10k in config)
  - cifar10     (60000 x 3072, capped to 10k in config)
"""

import os
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    fetch_openml,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def save_dataset(name, features, labels, class_suffix="Classes"):
    feat_path = os.path.join(OUT_DIR, f"{name}.csv")
    label_path = os.path.join(OUT_DIR, f"{name}{class_suffix}.csv")
    np.savetxt(feat_path, features, delimiter=",", fmt="%.8g")
    np.savetxt(label_path, labels, delimiter=",", fmt="%.0f")
    print(f"  {name}: {features.shape[0]} rows x {features.shape[1]} features -> {feat_path}")


def main():
    print("Downloading classic ML datasets...")

    # Iris
    d = load_iris()
    save_dataset("iris", d.data, d.target)

    # Wine
    d = load_wine()
    save_dataset("wine", d.data, d.target)

    # Breast cancer
    d = load_breast_cancer()
    save_dataset("cancer", d.data, d.target)

    # Digits
    d = load_digits()
    save_dataset("digits", d.data, d.target)

    print("\nDownloading image datasets (may take a minute)...")

    # MNIST (config expects data/mnistclasses.csv — lowercase)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    save_dataset("mnist", mnist.data.astype(float), mnist.target.astype(float), class_suffix="classes")

    # CIFAR-10 (config expects data/cifarClasses.csv — camelCase)
    cifar = fetch_openml("CIFAR_10", version=1, as_frame=False, parser="auto")
    save_dataset("cifar", cifar.data.astype(float), cifar.target.astype(float))

    print("\nDone!")


if __name__ == "__main__":
    main()
