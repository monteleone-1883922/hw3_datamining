from nltk import word_tokenize
import pandas as pd
import plotly.graph_objs as go
import sys
from enum import Enum
import cityhash
from plotly.subplots import make_subplots
import json
import numpy as np
import re

DATA_FILE_PATH = "Data/amazon_products_gpu.tsv"
SEED = 42
TOLLERANCE = 2e-3
MAX_ITERATIONS = 10
MAX_MEANS = 20
STOPWORDS_FILE = "Data/stopwords_list_it.json"
SPECIAL_CHARACTERS_FILE = "Data/special_characters.json"
K = 10
T = 35
COMPONENTS = 3


def print_elbow_curve(variances, num_means, names=[]):
    # Crea il grafico della curva dell'Elbow con Plotly
    fig = go.Figure()
    for i in range(len(variances)):
        fig.add_trace(go.Scatter(x=list(range(1, num_means + 1)), y=variances[i], mode='lines+markers',name='Elbow Curve ' + str(i) if names == [] else names[i]))

    # Aggiungi titoli e etichette
    fig.update_layout(
        title='Elbow Curve',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Variance Intra-Cluster')
    )

    # Mostra il grafico
    fig.show()


def print_runningtimes_and_elbow_curve(variances, runningtimes, num_means, title='Analysis Graphs', names=[]):
    fig = make_subplots(rows=2, cols=1)
    means = [i for i in range(1, num_means + 1)]
    for i in range(len(variances)):
        fig.add_trace(go.Scatter(x=means, y=variances[i], mode='lines+markers',
                                 name='Elbow Curve ' + str(i) if names == [] else names[i][0]), row=1, col=1)
        fig.add_trace(go.Scatter(x=means, y=runningtimes[i], mode='lines+markers',
                                 name='Running Times ' + str(i) if names == [] else names[i][1]), row=2, col=1)

    # Aggiungi titoli e etichette
    fig.update_layout(
        title=title,
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Variance Intra-Cluster'),
        xaxis2=dict(title='Number of Clusters (k)'),
        yaxis2=dict(title='Running Times')
    )

    # Mostra il grafico
    fig.show()


def print_runningtimes(runningtimes, num_means, names=[]):
    # Crea il grafico della curva dell'Elbow con Plotly
    fig = go.Figure()
    for i in range(len(runningtimes)):
        fig.add_trace(go.Scatter(x=list(range(1, num_means + 1)), y=runningtimes[i], mode='lines+markers',name='running time ' + str(i) if names == [] else names[i]))

    # Aggiungi titoli e etichette
    fig.update_layout(
        title='Runningtimes wrt number of clusters',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Runningtimes')
    )

    # Mostra il grafico
    fig.show()


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


def load_raw_data():
    with open(DATA_FILE_PATH, "r") as f:
        return f.readlines()[1:]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE_PATH, sep="\t")


def load_and_retype_data() -> pd.DataFrame:
    data = load_data()
    retype_dataframe(data)
    return data

def time_in_ms(time_format):
    try:
        return int(time_format)
    except:
        # Definisci il pattern regex per estrarre minuti, secondi e millisecondi
        pattern = re.compile(r'(?:(\d+)\s*m\s*)?(?:(\d+)\s*s\s*)?(\d+)\s*ms')

        # Cerca il pattern nel formato di tempo fornito
        match = pattern.match(time_format)

        if match:
            # Estrai i valori di minuti, secondi e millisecondi, gestendo i valori opzionali
            minuti = int(match.group(1)) if match.group(1) else 0
            secondi = int(match.group(2)) if match.group(2) else 0
            millisecondi = int(match.group(3))

            # Calcola il tempo totale in millisecondi
            tempo_totale_in_millisecondi = (minuti * 60 + secondi) * 1000 + millisecondi

            return tempo_totale_in_millisecondi
        else:
            # Restituisci un messaggio di errore se il formato non è valido
            raise ValueError("Il formato del tempo non è valido. Utilizzare il formato 'm s ms'.")




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


class Representation(Enum):
    """Enum for representation of the document"""
    TFIDF = "tfidf"
    MINWISEHASHING = "minwise"
    RAW = "raw"
    ONE_HOT_ENCODING = "hot"


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


def produce_name(normalize, centralize, pca):
    out = " normalize" if normalize else ""
    out += " centralize" if centralize else ""
    out += " pca" if pca else ""
    return out

