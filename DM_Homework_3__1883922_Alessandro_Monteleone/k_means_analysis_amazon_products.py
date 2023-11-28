from transformers import AutoTokenizer
import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import numpy as np

MODEL = "facebook/bart-base"
DATA_FILE_PATH = "amazon_products_gpu.tsv"
SEED = 42
TOLLERANCE = 2e-3
MAX_ITERATIONS = 10
MAX_MEANS = 40


def load_raw_data():
    with open(DATA_FILE_PATH, "r") as f:
        return f.readlines()


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE_PATH, sep="\t")


def load_and_retype_data() -> pd.DataFrame:
    data = load_data()
    retype_dataframe(data)
    return data


def retype_dataframe(df: pd.DataFrame) -> None:
    df["price"] = df["price"].apply(lambda x: x.replace(".", "").replace(",", ".") if pd.notna(x) else x)
    df["price"] = df["price"].astype(float)
    df["stars"] = df["stars"].apply(lambda x: x.replace(",", ".") if pd.notna(x) else x)
    df["stars"] = df["stars"].astype(float)
    df["num_reviews"] = df["num_reviews"].astype(str)
    df["num_reviews"] = df["num_reviews"].apply(lambda x: x.replace(".", "") if x != "nan" else "-1")
    df["num_reviews"] = df["num_reviews"].astype(int)
    df["num_reviews"] = df["num_reviews"].apply(lambda x: pd.NA if x == -1 else x)
    df["prime"] = df["prime"].astype(bool)


def produce_elbow_curve(data, num_means):
    result = []
    for i in range(num_means):
        kmeans = KMeans(init='k-means++', random_state=SEED, tol=TOLLERANCE, max_iter=MAX_ITERATIONS)
        kmeans.fit(data)
        result.append(kmeans.inertia_)
    return result


def print_elbow_curve(variances, num_means):
    # Crea il grafico della curva dell'Elbow con Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(num_means)), y=variances, mode='lines+markers'))

    # Aggiungi titoli e etichette
    fig.update_layout(
        title='Elbow Curve',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Variance Intra-Cluster')
    )

    # Mostra il grafico
    fig.show()

def add_padding(data):
    max_len = len(max(data, key=lambda x: len(x)))
    for i in range(len(data)):
        data[i] = data[i] + [0 for _ in range(max_len - len(data[i]))]




def process_raw_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = load_raw_data()
    for i in range(len(data)):
        data[i] = tokenizer.encode(data[i])
    add_padding(data)
    variances = produce_elbow_curve(data, MAX_MEANS)
    print_elbow_curve(variances, MAX_MEANS)


if __name__ == "__main__":
    np.random.seed(SEED)
    process_raw_data()
