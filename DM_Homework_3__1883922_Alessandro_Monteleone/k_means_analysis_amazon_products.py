from transformers import AutoTokenizer
import pandas as pd
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import numpy as np
import warnings, json
from nltk.tokenize import word_tokenize
from math import log2
import sys

DATA_FILE_PATH = "amazon_products_gpu.tsv"
SEED = 42
TOLLERANCE = 2e-3
MAX_ITERATIONS = 10
MAX_MEANS = 25
STOPWORDS_FILE = "stopwords_list_it.json"
SPECIAL_CHARACTERS_FILE = "special_characters.json"


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


def produce_elbow_curve(data, num_means):
    result = []
    for i in range(1, num_means + 1):
        print_progress_bar(i/num_means)
        kmeans = KMeans(init='k-means++', n_clusters=i, random_state=SEED, tol=TOLLERANCE, max_iter=MAX_ITERATIONS)
        kmeans.fit(data)
        result.append(kmeans.inertia_)
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
            id_word = words_index.get(word,-1)
            if id_word == -1:
                converted_words.append(idx)
                words_index[word] = idx
                idx += 1
            else:
                converted_words.append(id_word)
        converted_data.append(converted_words)

    adjust_representations(converted_data,idx)
    return converted_data

def adjust_and_convert_data(data,preprocessor):
    words_index = {}
    idx = 0
    converted_data = []
    for el in data:

        words = preprocessor.preprocess(el)
        converted_words = {}

        for word in words:
            id_word = words_index.get(word, (-1,-1))
            if id_word[0] == -1:
                converted_words[word] = (idx,1)
                words_index[word] = (idx,1)
                idx += 1
            else:
                old_value = converted_words.get(word,(id_word[0],0))
                converted_words[word] = (id_word[0],old_value[1] + 1)
                if old_value[1] == 0:
                    words_index[word] = (id_word[0], words_index[word][1] + 1)

        converted_data.append(converted_words)
    return adjust_and_apply_tfidf(converted_data,words_index,idx)


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
            new_representation[item[1][0]] = item[1][1] * log2(n/words_index[item[0]][1])
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
    barra = "[" + "=" * (blocchi_compilati-1) + ">" + " " * (lunghezza_barra - blocchi_compilati) + "]"
    sys.stdout.write(f"\r{barra} {percentuale * 100:.2f}% completo")
    sys.stdout.flush()

def apply_feature_engineering():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    preprocessor = SentencePreprocessing(STOPWORDS_FILE, SPECIAL_CHARACTERS_FILE)
    data = load_and_retype_data()
    result_data = []
    data["description"] = adjust_and_convert_data(data["description"].tolist(), preprocessor)
    for index, row in data.iterrows():

        result_data.append(row["description"] + [replace_if_null(row["price"], 0),
                                                  replace_if_null(row["prime"], 0),
                                                  replace_if_null(row["stars"], 0),
                                                  replace_if_null(row["num_reviews"], 0)])

    # add_padding(result_data)
    variances = produce_elbow_curve(result_data, MAX_MEANS)
    print_elbow_curve(variances, MAX_MEANS)


if __name__ == "__main__":
    np.random.seed(SEED)
    process_raw_data()
    # apply_feature_engineering()
