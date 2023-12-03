import json
import time
import plotly.graph_objs as go
import numpy as np
import math
from random import gauss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
from enum import Enum

# Experiment configurations
VALUES_FOR_N = [1000, 10000, 100000]
VALUES_FOR_K = [50, 100, 200]
REPORT_FILE = "experiment_results.json"
COMPRESSED_REPORT_FILE = "experiment_results_compressed.json"
COMPONENTS = 3
PRECISION = 16
SEED = 42
TOLERANCE = 2e-3
MAX_ITERATIONS = 10

class OperationalMode(Enum):
    """Enum for operational mode"""
    DO_EXPERIMENT = "experiment"
    COMBINE_RESULTS = "combine"
    REFORMAT_TIME = "time"
    PRINT_METRIC = "print"

def generate_dataset(n: int, k: int, d: int, s: float):
    """generate the set of points"""
    if PRECISION == 32:
        dtype = np.float32
    elif PRECISION == 16:
        dtype = np.float16
    else:
        dtype = float
    data = np.ndarray(shape=(k * n, d + k), dtype=dtype, order='F')
    # for row
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
    """handles the experiments and the storing of the results"""

    def __init__(self, report_file, report_compressed_file):
        # Initialize KMeans and PCA models, and report file paths
        self.kmeans = KMeans(init='k-means++', random_state=SEED, tol=TOLERANCE, max_iter=MAX_ITERATIONS)
        self.pca = PCA(random_state=SEED)
        self.report_file = report_file
        self.report_compressed_file = report_compressed_file
        # Create empty report files
        with open(report_file, "w") as report:
            json.dump({}, report, indent=4)
        with open(report_compressed_file, "w") as report:
            json.dump({}, report, indent=4)

    def set_pca(self, components):
        """Set the number of components for PCA"""
        self.pca = PCA(n_components=components, random_state=SEED)

    def set_kmeans(self, k):
        """Set the number of clusters for KMeans"""
        self.kmeans = KMeans(n_clusters=k, init='k-means++', random_state=SEED, tol=1e-3, max_iter=MAX_ITERATIONS)

    def store_results(self, results):
        # Store the experiment results in the report file
        self.store_results_file(results, self.report_file)

    def store_results_compressed(self, results):
        """Store compressed experiment results in the compressed report file"""
        self.store_results_file(results, self.report_compressed_file)

    def store_results_file(self, results, file):
        """Store results in a JSON file"""
        with open(file, "r+") as report_file:
            report = json.load(report_file)
            report_file.seek(0)
            key = self.parameters_to_key(results["parameters"])
            report[key] = results
            json.dump(report, report_file, indent=4)

    def parameters_to_key(self, parameters):
        """Convert experiment parameters to a string key"""
        result = ""
        for el in parameters.items():
            result += f"{el[0]}={el[1]}-"
        return result[:-1]

    def experiment(self, data, parameters,extended_report = False):
        """Perform the experiment"""
        k = parameters["k"]
        n = parameters["n"]
        correct_clusters = [[j + i for j in range(0, n * k, k)] for i in range(k)]
        #does the experiment with k-means only
        experiment_results = {"parameters": parameters, "n_components": self.pca.n_components}
        start_time = int(time.time() * 1000)
        self.kmeans.fit(data)
        end_time = int(time.time() * 1000) - start_time
        experiment_results["kmeans_running_time"] = end_time
        clusters = divide_clusters(self.kmeans.labels_.tolist())
        similarity = get_clusters_similarity(clusters, k, n)
        experiment_results["kmeans_similarity"] = similarity
        experiment_results["kmeans_cost"] = self.kmeans.inertia_

        if extended_report:
            experiment_results_extended = {"parameters": parameters, "n_components": self.pca.n_components}
            experiment_results_extended["kmeans_running_time"] = end_time
            experiment_results_extended["kmeans_similarity"] = similarity
            experiment_results_extended["kmeans_cost"] = self.kmeans.inertia_
            experiment_results_extended["kmeans_labels"] = self.kmeans.labels_.tolist()
            experiment_results_extended["kmeans_centers"] = self.kmeans.cluster_centers_.tolist()

        # does the experiment with k-means applied after PCA
        start_time = int(time.time() * 1000)
        self.pca.fit(data)
        data_transformed = self.pca.transform(data)
        end_time = int(time.time() * 1000) - start_time
        experiment_results["pca_running_time"] = end_time
        if extended_report:
            experiment_results_extended["pca_running_time"] = end_time
        del data
        start_time = int(time.time() * 1000)
        self.kmeans.fit(data_transformed)
        end_time = int(time.time() * 1000) - start_time
        experiment_results["kmeans_after_pca_running_time"] = end_time
        experiment_results["kmeans_pca_total_running_time"] = \
            experiment_results["kmeans_after_pca_running_time"] + \
            experiment_results["pca_running_time"]
        clusters = divide_clusters(self.kmeans.labels_.tolist())
        similarity = get_clusters_similarity(clusters, k, n)
        experiment_results["kmeans_pca_similarity"] = similarity
        experiment_results["kmeans_pca_cost"] = self.kmeans.inertia_
        if extended_report:
            experiment_results_extended["kmeans_after_pca_running_time"] = end_time
            experiment_results_extended["kmeans_pca_total_running_time"] = \
                experiment_results_extended["kmeans_after_pca_running_time"] + \
                experiment_results_extended["pca_running_time"]
            experiment_results_extended["kmeans_pca_similarity"] = similarity
            experiment_results_extended["kmeans_pca_cost"] = self.kmeans.inertia_
            experiment_results_extended["kmeans_pca_labels"] = self.kmeans.labels_.tolist()
            experiment_results_extended["kmeans_pca_centers"] = self.kmeans.cluster_centers_.tolist()
            self.store_results(experiment_results_extended)
        self.store_results_compressed(experiment_results)



def do_experiment():
    """Perform experiments for various configurations"""
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


def combine_results(file1, file2):
    """Combine two experiment result files"""
    with open(file2, "r") as f:
        ex2 = json.load(f)

    with open(file1, "r") as f1:
        ex1 = json.load(f1)
    for key in ex1.keys():
        ex1[key]["kmeans_running_time"] = convert_time_format(ex1[key]["kmeans_running_time"])
        ex1[key]["pca_running_time"] = convert_time_format(ex1[key]["pca_running_time"])
        ex1[key]["kmeans_after_pca_running_time"] = convert_time_format(ex1[key]["kmeans_after_pca_running_time"])
        ex1[key]["kmeans_pca_total_running_time"] = convert_time_format(ex1[key]["kmeans_pca_total_running_time"])
    for key in ex2.keys():
        ex1[key] = ex2[key]
    with open(file1, "w") as f2:
        json.dump(ex1, f2, indent=4)


def reformat_time(file):
    """Reformat time in experiment result file"""
    with open(file, "r") as f1:
        ex1 = json.load(f1)
    for key in ex1.keys():
        ex1[key]["kmeans_running_time"] = convert_time_format(ex1[key]["kmeans_running_time"])
        ex1[key]["pca_running_time"] = convert_time_format(ex1[key]["pca_running_time"])
        ex1[key]["kmeans_after_pca_running_time"] = convert_time_format(ex1[key]["kmeans_after_pca_running_time"])
        ex1[key]["kmeans_pca_total_running_time"] = convert_time_format(ex1[key]["kmeans_pca_total_running_time"])
    with open(file, "w") as f2:
        json.dump(ex1, f2, indent=4)


def convert_time_format(milliseconds):
    """Convert time from milliseconds to a readable format"""
    if milliseconds >= 1000:
        seconds = milliseconds // 1000
        milliseconds_left = milliseconds % 1000

        if seconds < 60:
            return f"{seconds} s {milliseconds_left} ms"
        else:
            minutes = seconds // 60
            seconds_left = seconds % 60
            return f"{minutes} m {seconds_left} s {milliseconds_left} ms"
    return f"{milliseconds} ms"


def divide_clusters(clusters):
    """Divide clusters into subclusters"""
    divided_clusters = {}
    for i in range(len(clusters)):
        divided_clusters[clusters[i]] = divided_clusters.get(clusters[i], []) + [i]
    return [value for value in divided_clusters.values()]


def get_clusters_similarity(clusters, k, n):
    """Calculate the similarity between clusters"""
    total_similarity = 0
    for i in range(k):
        correct_cluster = [j + i for j in range(0, n * k, k)]
        jaccard_similarity = -1
        for cluster in clusters:
            jaccard_similarity = max(compute_jaccard_similarity(set(correct_cluster), set(cluster)), jaccard_similarity)
        total_similarity += jaccard_similarity
    return total_similarity / k


def compute_jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets"""
    intersection_sets = set1.intersection(set2)
    union_sets = set1.union(set2)
    return len(intersection_sets) / len(union_sets)


def load_experiment_results(file):
    """Load experiment results from a file"""
    new_data = []
    with open(file, "r") as f:
        data = json.load(f)
    for key in data.keys():
        new_data.append(data[key])
    return data


def print_graph(data, type_metric):
    """Print a graph based on experiment results"""
    x = []
    y = []
    for key in data.keys():
        x.append(key)
        y.append(data[key][type_metric])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))

    # Add titles and labels
    fig.update_layout(
        title='Analysis',
        xaxis=dict(title='parameters'),
        yaxis=dict(title=type_metric)
    )

    # Show the graph
    fig.show()


def main():
    """Main function to execute based on command line arguments"""
    if len(sys.argv) < 2:
        print("missing input arguments", sys.stderr)
        exit(1)
    #does the experiment
    if sys.argv[1].lower().find(OperationalMode.DO_EXPERIMENT.value) != -1:
        np.random.seed(SEED)
        do_experiment()
    #combine results in 2 different result files
    if sys.argv[1].lower().find(OperationalMode.COMBINE_RESULTS.value) != -1:
        if len(sys.argv) < 4:
            print("missing input arguments file1 and file2", sys.stderr)
            exit(1)
        combine_results(sys.argv[2], sys.argv[3])
    #change time format in a result file
    if sys.argv[1].lower().find(OperationalMode.REFORMAT_TIME.value) != -1:
        if len(sys.argv) < 3:
            print("missing input argument file", sys.stderr)
            exit(1)
        reformat_time(sys.argv[2])
    #prints a graph showing the changes of a metric through the different configurations
    if sys.argv[1].lower().find(OperationalMode.PRINT_METRIC.value) != -1:
        if len(sys.argv) < 4:
            print("missing input arguments file and metric", sys.stderr)
            exit(1)
        data = load_experiment_results(sys.argv[2])
        print_graph(data, sys.argv[3])


if __name__ == "__main__":
    main()
