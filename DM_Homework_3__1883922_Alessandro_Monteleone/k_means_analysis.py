import json
import time

import numpy as np
import math, sys
from random import gauss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

VALUES_FOR_N = [1000, 10000, 100000]
VALUES_FOR_K = [50, 100, 200]
REPORT_FILE = "experiment_results.json"
COMPRESSED_REPORT_FILE = "experiment_results_compressed.json"
COMPONENTS = 3
PRECISION = 16
SEED = 42
TOLLERANCE = 2e-3
MAX_ITERATIONS = 10


def generate_dataset(n: int, k: int, d: int, s: float):
    if PRECISION == 32:
        dtype = np.float32
    elif PRECISION == 16:
        dtype = np.float16
    else:
        dtype = float
    data = np.ndarray(shape=(k * n, d + k), dtype=dtype, order='F')
    # for riga
    for i in range(k * n):
        # for el in gauss matrix
        for j in range(d):
            data[i][k + j] = gauss(0, s)
        # for el in identity matrix
        for j in range(k):
            if i % k == j:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data


class Report():

    def __init__(self, report_file, report_compressed_file):
        self.kmeans = KMeans(init='k-means++', random_state=SEED, tol=TOLLERANCE, max_iter=MAX_ITERATIONS)
        self.pca = PCA(random_state=SEED)
        self.report_file = report_file
        self.report_compressed_file = report_compressed_file
        with open(report_file, "w") as report:
            json.dump({}, report, indent=4)
        with open(report_compressed_file, "w") as report:
            json.dump({}, report, indent=4)

    def set_pca(self, components):
        self.pca = PCA(n_components=components, random_state=SEED)

    def set_kmeans(self, k):
        self.kmeans = KMeans(n_clusters=k, init='k-means++', random_state=SEED, tol=1e-3, max_iter=MAX_ITERATIONS)

    def store_results(self, results):
        self.store_results_file(results, self.report_file)

    def store_results_compressed(self, results):
        self.store_results_file(results, self.report_compressed_file)

    def store_results_file(self, results, file):
        with open(file, "r+") as report_file:
            report = json.load(report_file)
            report_file.seek(0)
            key = self.parameters_to_key(results["parameters"])
            report[key] = results
            json.dump(report, report_file, indent=4)

    def parameters_to_key(self, parameters):
        result = ""
        for el in parameters.items():
            result += f"{el[0]}={el[1]}-"
        return result[:-1]

    def experiment(self, data, parameters):
        k = parameters["k"]
        n = parameters["n"]
        correct_clusters = [[j + i for j in range(0, n * k, k)] for i in range(k)]
        experiment_results = compressed_results = {"parameters": parameters, "n_components": self.pca.n_components}
        start_time = int(time.time() * 1000)
        self.kmeans.fit(data)
        experiment_results["kmeans_running_time"] = compressed_results["kmeans_running_time"] = int(
            time.time() * 1000) - start_time
        clusters = divide_clusters(self.kmeans.labels_.tolist())
        experiment_results["kmeans_similarity"] = compressed_results["similarity"] = get_clusters_similarity(clusters,
                                                                                                             k, n)
        experiment_results["kmeans_cost"] = compressed_results["kmeans_cost"] = self.kmeans.inertia_
        experiment_results["kmeans_labels"] = self.kmeans.labels_.tolist()
        experiment_results["kmeans_centers"] = self.kmeans.cluster_centers_.tolist()
        start_time = int(time.time() * 1000)
        self.pca.fit(data)
        data_transformed = self.pca.transform(data)
        experiment_results["pca_running_time"] = int(time.time() * 1000) - start_time
        del data
        start_time = int(time.time() * 1000)
        self.kmeans.fit(data_transformed)
        experiment_results["kmeans_after_pca_running_time"] = int(time.time() * 1000) - start_time
        experiment_results["kmeans_pca_total_running_time"] = experiment_results["kmeans_after_pca_running_time"] + \
                                                              experiment_results["pca_running_time"]
        clusters = divide_clusters(self.kmeans.labels_.tolist())
        experiment_results["kmeans_similarity"] = compressed_results["similarity"] = get_clusters_similarity(clusters,
                                                                                                             k, n)
        experiment_results["kmeans_cost"] = compressed_results["kmeans_cost"] = self.kmeans.inertia_
        experiment_results["kmeans_pca_labels"] = self.kmeans.labels_.tolist()
        experiment_results["kmeans_pca_centers"] = self.kmeans.cluster_centers_.tolist()

        self.store_results_compressed(compressed_results)
        self.store_results(experiment_results)


def combine_parameters():
    report = Report(REPORT_FILE, COMPRESSED_REPORT_FILE)
    for n in VALUES_FOR_N:
        for k in VALUES_FOR_K:
            report.set_kmeans(k)

            values_for_d = [k, 100 * k]
            values_for_s = [1 / k, 1 / math.sqrt(k), 0.5]
            for d in values_for_d:
                report.set_pca(k // COMPONENTS)
                for s in values_for_s:
                    parameters = {
                        "n": n,
                        "k": k,
                        "d": d,
                        "s": s
                    }
                    print(
                        f"generating dataset for \nn = {n}\nk = {k}\nd = {d}\ns = {s}\nnum component is {(k + d) // COMPONENTS} ")
                    data = generate_dataset(n, k, d, s)
                    print("start experiment")
                    report.experiment(data, parameters)


def combine_results():
    with open("experiment_results2.json") as f:
        ex2 = json.load(f)

    with open("experiment_results.json", "r") as f1:
        ex1 = json.load(f1)
    for key in ex2.keys():
        ex1[key] = ex2[key]
    with open("experiment_results.json", "w") as f2:
        json.dump(ex1, f2)


def divide_clusters(clusters):
    divided_clusters = {}
    for i in range(len(clusters)):
        divided_clusters[clusters[i]] = divided_clusters.get(clusters[i], []) + [i]
    return [value for value in divided_clusters.values()]


def get_clusters_similarity(clusters, k, n):
    total_similarity = 0
    for i in range(k):
        correct_cluster = [j + i for j in range(0, n * k, k)]
        jaccard_similarity = -1
        for cluster in clusters:
            jaccard_similarity = max(compute_jaccard_similarity(set(correct_cluster), set(cluster)), jaccard_similarity)
        total_similarity += jaccard_similarity
    return total_similarity / k


def compute_jaccard_similarity(set1, set2):
    intersection_sets = set1.intersection(set2)
    union_sets = set1.union(set2)
    return len(intersection_sets) / len(union_sets)


if __name__ == "__main__":
    # combine_results()
    np.random.seed(SEED)
    combine_parameters()
