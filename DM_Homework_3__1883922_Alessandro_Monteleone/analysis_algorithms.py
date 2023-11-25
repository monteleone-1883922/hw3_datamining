import numpy as np
from random import randint, random
from sklearn.cluster import KMeans


def pca(data, m: int):
    u, s, vh = np.linalg.svd(data, full_matrices=True)
    smat = np.zeros(data.shape, dtype=float)
    smat[:s.size, :s.size] = np.diag(s)
    for i in range(0, s.size):
        if i > m - 1:
            s[i] = 0
    smat[:data.shape[1], :data.shape[1]] = np.diag(s)
    projected = np.dot(u, np.dot(smat, vh))
    a = 2


def compute_probabilities(points, means):
    total_prob = 0
    probabilities = [0 for _ in points]
    for i in range(len(points)):
        point = points[i]
        p = -1
        for mean in means:
            p = p if p == -1 else min(np.sqrt(np.sum((point - mean) ** 2)), p)
        total_prob += p
        probabilities[i] = p
    probabilities[0] /= total_prob
    for i in range(1, len(probabilities)):
        probabilities[i] = probabilities[i - 1] + probabilities[i] / total_prob
    return probabilities


def binary_search(search_list, element):
    idx = len(search_list) // 2
    while True:
        if element > search_list[idx]:
            if element <= search_list[idx + 1]:
                return idx + 1
            idx += idx // 2
        else:
            if element > search_list[idx - 1]:
                return idx
            idx -= idx // 2


def pickup_random_point(points, probabilities):
    r = random()
    idx = binary_search(probabilities, r)
    return points[idx]


def k_means_pp(data, k):
    means = [data[randint(0, data.shape[0])]]
    for i in range(k - 1):
        probabilities = compute_probabilities(data, means)
        mean = pickup_random_point(data, probabilities)
        means.append(mean)
    return means
