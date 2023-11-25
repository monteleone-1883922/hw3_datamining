import numpy as np
import math, sys
from random import gauss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

VALUES_FOR_N = [20, 1000, 10000, 100000]
VALUES_FOR_K = [5,50, 100, 200]


def generate_dataset(n: int, k: int, d: int, s: float):
    data = np.ndarray(shape=(k * n, d + k), dtype=float, order='F')
    # for riga
    for i in range(k * n):
        # for el in gaus matrix
        for j in range(d):
            data[i][k + j] = gauss(0, s)
        # for el in identity matrix
        for j in range(k):
            if i % k == j:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


def combine_parameters():
    for n in VALUES_FOR_N:
        for k in VALUES_FOR_K:
            values_for_d = [k, 100 * k, 100 * k ** 2]
            values_for_s = [1 / k, 1 / math.sqrt(k), 0.5]
            for d in values_for_d:
                for s in values_for_s:
                    print(f"generating dataset for \nn = {n}\nk = {k}\nd = {d}\ns = {s}")
                    data = generate_dataset(n, k, d, s)


if __name__ == "__main__":
    combine_parameters()