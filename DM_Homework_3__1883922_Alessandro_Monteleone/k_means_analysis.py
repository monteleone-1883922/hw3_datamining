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
COMPONENTS = 3
PRECISION = 16



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

    def __init__(self, report_file):
        self.kmeans = KMeans(init='k-means++')
        self.pca = PCA()
        self.report_file = report_file
        with open(report_file, "w") as report:
            json.dump({}, report)

    def set_pca(self, components):
        self.pca = PCA(n_components=components)

    def set_kmeans(self, k):
        self.kmeans = KMeans(n_clusters=k, init='k-means++')

    def store_results(self, results):
        with open(self.report_file, "r+") as report_file:
            report = json.load(report_file)
            report_file.seek(0)
            key = self.parameters_to_key(results["parameters"])
            report[key] = results
            json.dump(report, report_file)

    def parameters_to_key(self, parameters):
        result = ""
        for el in parameters.items():
            result += f"{el[0]}={el[1]}-"
        return result[:-1]

    def experiment(self, data, parameters):
        experiment_results = {"parameters": parameters, "n_components": self.pca.n_components}
        start_time = int(time.time() * 1000)
        self.pca.fit(data)
        data_transformed = self.pca.transform(data)
        experiment_results["pca_running_time"] = int(time.time() * 1000) - start_time
        start_time = int(time.time() * 1000) - start_time
        self.kmeans.fit(data_transformed)
        experiment_results["kmeans_after_pca_running_time"] = int(time.time() * 1000) - start_time
        experiment_results["kmeans_pca_total_running_time"] = experiment_results["kmeans_after_pca_running_time"] + \
                                                              experiment_results["pca_running_time"]
        experiment_results["kmeans_pca_labels"] = self.kmeans.labels_.tolist()
        experiment_results["kmeans_pca_centers"] = self.kmeans.cluster_centers_.tolist()
        start_time = int(time.time() * 1000) - start_time
        self.kmeans.fit(data)
        experiment_results["kmeans_running_time"] = int(time.time() * 1000) - start_time
        experiment_results["kmeans_labels"] = self.kmeans.labels_.tolist()
        experiment_results["kmeans_centers"] = self.kmeans.cluster_centers_.tolist()
        self.store_results(experiment_results)


def combine_parameters():
    report = Report(REPORT_FILE)
    for n in VALUES_FOR_N:
        for k in VALUES_FOR_K:
            report.set_kmeans(k)

            values_for_d = [k, 100 * k, 100 * k ** 2]
            values_for_s = [1 / k, 1 / math.sqrt(k), 0.5]
            for d in values_for_d:
                report.set_pca((k+d)//COMPONENTS)
                for s in values_for_s:
                    parameters = {
                        "n": n,
                        "k": k,
                        "d": d,
                        "s": s
                    }
                    print(f"generating dataset for \nn = {n}\nk = {k}\nd = {d}\ns = {s}\nnum component is {(k+d)//COMPONENTS} ")
                    data = generate_dataset(n, k, d, s)
                    report.experiment(data,parameters)



if __name__ == "__main__":
    np.random.seed(42)
    combine_parameters()
