import sys
import time
from sklearn.cluster import KMeans
import warnings
from math import log2
from sklearn.decomposition import PCA
from utils import *


def perform_clusterization(data, num_means, clusterization=False, cluster=-1):
    if cluster > 0:
        kmeans = KMeans(init='k-means++', n_clusters=cluster, random_state=SEED, tol=TOLLERANCE,
                        max_iter=MAX_ITERATIONS)
        kmeans.fit(data)
        return kmeans.labels_.tolist()
    result = []
    runningtimes = []
    clusterization_list = []
    for i in range(1, num_means + 1):
        print_progress_bar(i / num_means)
        kmeans = KMeans(init='k-means++', n_clusters=i, random_state=SEED, tol=TOLLERANCE, max_iter=MAX_ITERATIONS)
        start_time = int(time.time() * 1000)
        kmeans.fit(data)
        runningtime = int(time.time() * 1000) - start_time
        if clusterization:
            clusterization_list.append(kmeans.cluster_centers_.tolist())

        runningtimes.append(runningtime)
        result.append(kmeans.inertia_)

    return result, runningtimes, clusterization_list


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

    add_padding(converted_data, idx)
    return converted_data


def add_padding(data, max_len):
    for i in range(len(data)):
        data[i] = data[i] + [0 for _ in range(max_len - len(data[i]))]


def one_hot_encodings(data, preprocessor):
    words_index = {}
    idx = 0
    converted_data = []
    for el in data:
        words = preprocessor.preprocess(el)
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


def process_raw_data(data=None, normalize=False, clusterization_only=False, cluster=-1):
    # Ignora i FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if data is None:
        data = load_raw_data()
    data = convert_raw_data_into_numbers(data)
    if normalize:
        matrix_norma = np.linalg.norm(np.array(data))
        data = np.array(data) / matrix_norma
    if cluster > 0:
        return perform_clusterization(data, MAX_MEANS, cluster=cluster)
    variances, runningtimes, clusters = perform_clusterization(data, MAX_MEANS)
    if clusterization_only:
        return clusters
    return variances, runningtimes


def minwisehashing_representation(data, shingling, minwisehashing, preprocessor):
    new_representation = []
    for el in data:
        preprocessed_el = " ".join(preprocessor.preprocess(el))
        shigling_format = shingling.get_shingling(preprocessed_el)
        minwisehashing_format = minwisehashing.get_minwise_hashing(shigling_format)
        new_representation.append(minwisehashing_format)
    return new_representation


def apply_feature_engineering(representation, data=None, clusterization_only=False, normalize=False, centralize=False,
                              pca=False, components=COMPONENTS, cluster=-1, additional_fields=[]):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    preprocessor = SentencePreprocessing(STOPWORDS_FILE, SPECIAL_CHARACTERS_FILE)
    start_time = int(time.time() * 1000)
    if data is None:
        data = load_and_retype_data()
        data = data.drop_duplicates(subset='description', keep='first')
    result_data = []
    if representation == Representation.TFIDF:
        data["description"] = adjust_and_convert_data(data["description"].tolist(), preprocessor)
    elif representation == Representation.MINWISEHASHING:
        shingling_hash = hashFamily(1)
        shingling = Shingling(shingling_hash, K)
        minwisehash_functions = [hashFamily(i) for i in range(T)]
        minwisehashing = MinwiseHashing(minwisehash_functions, T)
        data["description"] = minwisehashing_representation(data["description"].tolist(), shingling, minwisehashing,
                                                            preprocessor)
    elif representation == Representation.ONE_HOT_ENCODING:
        data["description"] = one_hot_encodings(data["description"].tolist(), preprocessor)
    central_vec = None
    for index, row in data.iterrows():
        vec = np.array(row["description"] + [replace_if_null(row[field], 0) for field in additional_fields])
        """replace_if_null(row["price"], 0),
                                             replace_if_null(row["prime"], 0),
                                             replace_if_null(row["stars"], 0),
                                             replace_if_null(row["num_reviews"], 0)"""
        result_data.append(vec)
        # central_vec = vec / data.shape[0] if central_vec is None else central_vec + vec / data.shape[0]
    # normalize
    if normalize:
        matrix_norma = np.linalg.norm(np.array(result_data))
        result_data = np.array(result_data) / matrix_norma
        # central_vec = central_vec / matrix_norma
    # centralize
    # if centralize:
    #     for i in range(len(result_data)):
    #         result_data[i] = result_data[i] - central_vec
    end_time = int(time.time() * 1000) - start_time
    if pca:
        pca_obj = PCA(n_components=components, random_state=SEED)
        pca_obj.fit(result_data)
        result_data = pca_obj.transform(result_data)

    if cluster > 0:
        return perform_clusterization(result_data, MAX_MEANS, cluster=cluster)
    variances, runningtimes, clusters = perform_clusterization(result_data, MAX_MEANS)
    if clusterization_only:
        return clusters
    return variances, runningtimes


def main():
    if len(sys.argv) < 4:
        print(
            "arguments missing, you have to specify what to plot (running times or elbow curve or both), what techniques to use and the max number of means",
            sys.stderr)
        exit(1)
    np.random.seed(SEED)
    raw_techniques = sys.argv[2].split(",")
    techniques = []

    for i in range(len(raw_techniques)):
        normalize = raw_techniques[i].find("normalize") != -1
        if raw_techniques[i].find(Representation.RAW.value) != -1:
            techniques.append((Representation.RAW, normalize, None))

        else:
            pca = raw_techniques[i].find("pca") != -1
            if raw_techniques[i].find(Representation.TFIDF.value) != -1:
                techniques.append((Representation.TFIDF, normalize, pca))
            elif raw_techniques[i].find(Representation.MINWISEHASHING.value) != -1:
                techniques.append((Representation.MINWISEHASHING, normalize, pca))
            elif raw_techniques[i].find(Representation.ONE_HOT_ENCODING.value) != -1:
                techniques.append((Representation.ONE_HOT_ENCODING, normalize, pca))
            else:
                print("technique not found", sys.stderr)
                exit(1)

    variances = []
    running_times = []
    names = []
    if sys.argv[1].find("elbow") != -1 and sys.argv[1].find("time") != -1:

        for technique in techniques:
            if technique[0] == Representation.RAW:
                raw_variances, raw_runningtimes = process_raw_data(normalize=technique[1])
                variances.append(raw_variances)
                running_times.append(raw_runningtimes)
                names.append(("raw data variance", "raw data running time"))
            else:
                feature_variances, feature_runningtimes = apply_feature_engineering(technique[0], normalize=technique[1],
                                                                                    pca=technique[2])
                variances.append(feature_variances)
                running_times.append(feature_runningtimes)
                names.append((technique[0].name.lower() + " variance", technique[0].name.lower() + " running time"))
        print_runningtimes_and_elbow_curve(variances, running_times, int(sys.argv[3]), names=names)

    elif sys.argv[1].find("elbow") != -1:
        for technique in techniques:
            if technique[0] == Representation.RAW:
                raw_variances, raw_runningtimes = process_raw_data(normalize=technique[1])
                variances.append(raw_variances)
                names.append("raw data variance")
            else:
                feature_variances, feature_runningtimes = apply_feature_engineering(technique, normalize=technique[1],
                                                                                    pca=technique[2])
                variances.append(feature_variances)
                names.append(technique[0].name.lower() + " variance")
        print_elbow_curve(variances, int(sys.argv[3]), names=names)


    elif sys.argv[1].find("time") != -1:
        for technique in techniques:
            if technique[0] == Representation.RAW:
                raw_variances, raw_runningtimes = process_raw_data(normalize=technique[1])
                running_times.append(raw_runningtimes)
                names.append("raw data running time")
            else:
                feature_variances, feature_runningtimes = apply_feature_engineering(technique, normalize=technique[1],
                                                                                    pca=technique[2])
                running_times.append(feature_runningtimes)
                names.append(technique[0].name.lower() + " running time")
        print_runningtimes(running_times, int(sys.argv[3]), names=names)


if __name__ == "__main__":
    main()

