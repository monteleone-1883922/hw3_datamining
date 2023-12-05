import time
from enum import Enum

from transformers import AutoTokenizer
import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import numpy as np
import warnings, json
from nltk.tokenize import word_tokenize
from math import log2
import sys
import cityhash

DATA_FILE_PATH = "amazon_products_gpu.tsv"
SEED = 42
TOLLERANCE = 2e-3
MAX_ITERATIONS = 10
MAX_MEANS = 25
STOPWORDS_FILE = "stopwords_list_it.json"
SPECIAL_CHARACTERS_FILE = "special_characters.json"
K = 10
T = 35


class Representation(Enum):
    """Enum for representation of the document"""
    TFIDF = "tfidf"
    MINWISEHASHING = "minwisehashing"


class SentencePreprocessing():

    def __init__(self, stopwords_file_path: str, special_characters_file_path: str):

        with open(stopwords_file_path, 'r') as stopwords_file:
            data = json.load(stopwords_file)
        self.stopwords = set(data["words"])
        with open(special_characters_file_path, 'r') as special_characters_file:
            data = json.load(special_characters_file)
        self.special_characters = set(data["special_characters"])

    def remove_stopwords(self, words: list[str]) -> list[str]:
        result = []
        for word in words:
            if word.lower() not in self.stopwords and word not in self.special_characters:
                result.append(word.lower())
        return result

    def remove_special_characters(self, words: list[str]) -> list[str]:
        result = []
        for word in words:
            if word not in self.special_characters:
                result.append(word.lower())
        return result

    def preprocess(self, sentence: str, remove_stopwords: bool = True):
        tokenized = word_tokenize(sentence)
        return self.remove_stopwords(tokenized) if remove_stopwords else self.remove_special_characters(
            tokenized)


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


def produce_elbow_curve(data, num_means, save_runningtimes=False):
    result = []
    runningtimes = []
    for i in range(1, num_means + 1):
        print_progress_bar(i / num_means)
        kmeans = KMeans(init='k-means++', n_clusters=i, random_state=SEED, tol=TOLLERANCE, max_iter=MAX_ITERATIONS)
        start_time = int(time.time() * 1000)
        kmeans.fit(data)
        runningtime = int(time.time() * 1000) - start_time
        if save_runningtimes:
            runningtimes.append(runningtime)
        result.append(kmeans.inertia_)

    if save_runningtimes:
        return result, runningtimes
    return result


def print_elbow_curve(variances, num_means):
    # Crea il grafico della curva dell'Elbow con Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_means + 1)), y=variances, mode='lines+markers'))

    # Aggiungi titoli e etichette
    fig.update_layout(
        title='Elbow Curve',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Variance Intra-Cluster')
    )

    # Mostra il grafico
    fig.show()


def print_runningtimes(runningtimes, num_means):
    # Crea il grafico della curva dell'Elbow con Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, num_means + 1)), y=runningtimes, mode='lines+markers'))

    # Aggiungi titoli e etichette
    fig.update_layout(
        title='Runningtimes wrt number of clusters',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Runningtimes')
    )

    # Mostra il grafico
    fig.show()


def add_padding(data):
    max_len = len(max(data, key=lambda x: len(x)))
    for i in range(len(data)):
        data[i] = data[i] + [0 for _ in range(max_len - len(data[i]))]


def convert_raw_data_into_numbers(data):
    words_index = {}
    idx = 0
    converted_data = []
    for el in data:

        el1 = el.replace("\t", " ")
        words = word_tokenize(el1)
        converted_words = []

        for word in words:
            id_word = words_index.get(word, -1)
            if id_word == -1:
                converted_words.append(idx)
                words_index[word] = idx
                idx += 1
            else:
                converted_words.append(id_word)
        converted_data.append(converted_words)

    adjust_representations(converted_data, idx)
    return converted_data


def adjust_and_convert_data(data, preprocessor):
    words_index = {}
    idx = 0
    converted_data = []
    for el in data:

        words = preprocessor.preprocess(el)
        converted_words = {}

        for word in words:
            id_word = words_index.get(word, (-1, -1))
            if id_word[0] == -1:
                converted_words[word] = (idx, 1)
                words_index[word] = (idx, 1)
                idx += 1
            else:
                old_value = converted_words.get(word, (id_word[0], 0))
                converted_words[word] = (id_word[0], old_value[1] + 1)
                if old_value[1] == 0:
                    words_index[word] = (id_word[0], words_index[word][1] + 1)

        converted_data.append(converted_words)
    return adjust_and_apply_tfidf(converted_data, words_index, idx)


def adjust_representations(representation, max_len):
    for i in range(len(representation)):
        new_representation = [0 for _ in range(max_len)]
        for id in representation[i]:
            new_representation[id] = 1
        representation[i] = new_representation


def adjust_and_apply_tfidf(data, words_index, max_len):
    new_data = []
    n = len(data)
    for el in data:
        new_representation = [0 for _ in range(max_len)]
        for item in el.items():
            new_representation[item[1][0]] = item[1][1] * log2(n / words_index[item[0]][1])
        new_data.append(new_representation)
    return new_data


def process_raw_data():
    # Ignora i FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    data = load_raw_data()
    data = convert_raw_data_into_numbers(data)

    variances = produce_elbow_curve(data, MAX_MEANS)
    print_elbow_curve(variances, MAX_MEANS)


def replace_if_null(value, replace):
    if value is None or pd.isna(value) or np.isnan(value):
        return replace
    return value


def print_progress_bar(percentuale, lunghezza_barra=20):
    blocchi_compilati = int(lunghezza_barra * percentuale)
    barra = "[" + "=" * (blocchi_compilati - 1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% completo")
    sys.stdout.flush()


class Shingling():

    def __init__(self, hash_function, k: int):
        self.hash = hash_function
        self.k = k

    def get_shingling(self, document):
        set_of_shinglings = []

        for i in range(len(document) - self.k + 1):
            word = document[i: i + self.k]
            set_of_shinglings.append(self.hash(word))
        return set_of_shinglings


def hashFamily(i):
    resultSize = 8
    # how many bytes we want back
    maxLen = 20
    # how long can our i be (in decimal)
    salt = str(i).zfill(maxLen)[-maxLen:]
    salt = salt.encode('utf-8')

    def hashMember(x):
        if type(x) is str:
            x = x.encode('utf-8')
        elif type(x) is list:
            x = b"".join(x)
        elif type(x) is int:
            x = str(x).encode('utf-8')
        return cityhash.CityHash32(x + salt)

    return hashMember


class MinwiseHashing():

    def __init__(self, hash_functions, t: int):
        self.hashes = hash_functions
        self.t = t

    def get_minwise_hashing(self, elements_set):
        set_signature = []
        for i in range(self.t):
            min_hash = None
            for el in elements_set:
                min_hash = min(min_hash, self.apply_hash(el, i)) if not min_hash is None else self.apply_hash(el, i)
            set_signature.append(min_hash)
        return set_signature

    def apply_hash(self, el, i):
        hash = self.hashes[i % len(self.hashes)]
        return hash(el)


def minwisehashing_representation(data, shingling, minwisehashing, preprocessor):
    new_representation = []
    for el in data:
        preprocessed_el = " ".join(preprocessor.preprocess(el))
        shigling_format = shingling.get_shingling(preprocessed_el)
        minwisehashing_format = minwisehashing.get_minwise_hashing(shigling_format)
        new_representation.append(minwisehashing_format)
    return new_representation


def apply_feature_engineering(representation, normalize=False, centralize=False):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    preprocessor = SentencePreprocessing(STOPWORDS_FILE, SPECIAL_CHARACTERS_FILE)
    data = load_and_retype_data()
    result_data = []
    if representation == Representation.TFIDF:
        data["description"] = adjust_and_convert_data(data["description"].tolist(), preprocessor)
    elif representation == Representation.MINWISEHASHING:
        shingling_hash = hashFamily(1)
        shingling = Shingling(shingling_hash, K)
        minwisehash_functions = [hashFamily(i) for i in range(T)]
        minwisehashing = MinwiseHashing(minwisehash_functions, T)
        data["description"] = minwisehashing_representation(data["description"].tolist(), shingling, minwisehashing, preprocessor)
    central_vec = None
    for index, row in data.iterrows():
        vec = np.array(row["description"] + [replace_if_null(row["price"], 0),
                                             replace_if_null(row["prime"], 0),
                                             replace_if_null(row["stars"], 0),
                                             replace_if_null(row["num_reviews"], 0)])
        result_data.append(vec)
        central_vec = vec / data.shape[0] if central_vec is None else central_vec + vec / data.shape[0]
    # normalize
    if normalize:
        matrix_norma = np.linalg.norm(np.array(result_data))
        result_data = np.array(result_data) / matrix_norma
        central_vec = central_vec / matrix_norma
    # centralize
    if centralize:
        for i in range(len(result_data)):
            result_data[i] = result_data[i] - central_vec
    variances = produce_elbow_curve(result_data, MAX_MEANS)
    print_elbow_curve(variances, MAX_MEANS)


if __name__ == "__main__":
    np.random.seed(SEED)
    # process_raw_data()
    apply_feature_engineering(Representation.MINWISEHASHING, normalize=True)
