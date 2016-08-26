import numpy as np
import re
import itertools
from collections import Counter
import ujson as json
import joblib

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_text_word_splits(training_data):
    """ Get each line in the data as individual items in a list and the split each sentence on spaces. """
    text = [clean_str(i) for i in training_data]
    text = [s.split(" ") for s in text]
    return text

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    model_training_data = "/raid/ankit/lstm/model_training_data.txt"
    print "Loading Model Training Data: {}".format(model_training_data)

    training_data_query = []
    training_data_similar = []
    training_data_non_similar1 = []
    training_data_non_similar2 = []
    training_data_non_similar3 = []

    # output = {'q': query, 'doc_corr': correct_url_doc, 'doc_incorr': incorrect_doc_list}
    less_doc_cnt = 0
    with open(model_training_data) as fo:

        for line in fo:

            data = json.loads(line)
            if len(data['doc_incorr']) == 3:
                training_data_query.append(data['q'])
                training_data_similar.append(data['doc_corr'])

                training_data_non_similar1.append(data['doc_incorr'][0])
                training_data_non_similar2.append(data['doc_incorr'][1])
                training_data_non_similar3.append(data['doc_incorr'][2])
            else:
                less_doc_cnt +=1
    print "Number of skipped data points: Incorrect Documents in Training Data (< 3): {}".format(less_doc_cnt)

    # Split by words

    query_text = get_text_word_splits(training_data_query)

    similar_text = get_text_word_splits(training_data_similar)

    nonsimilar_text1 = get_text_word_splits(training_data_non_similar1)

    nonsimilar_text2 = get_text_word_splits(training_data_non_similar2)

    nonsimilar_text3 = get_text_word_splits(training_data_non_similar3)

    # Generate labels
    data_output_labels = [1 for _ in similar_text]

    # y = np.concatenate([positive_labels, negative_labels], 0)
    return [ query_text, similar_text, nonsimilar_text1, nonsimilar_text2, nonsimilar_text3 , np.asarray(data_output_labels)]



def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)

    print "Maximum Length of the Sequence: {}".format(sequence_length)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word, HAS list of words
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index, HAS A Dict where key is word and value is the index position or a unique identifier which corresponds to the word.
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary_inv


def build_input_data(sentences, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary.get(word) for word in sentence] for sentence in sentences])
    return x

def load_word_embeddings(embedding_dim, vocab_list, masking=False, use_pickled=True):

    print("Loading Word Embeddings into memory ... ")
    if masking:
        masking_value = ""  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
    else:
        masking_value = "_non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"

    # Dataset sources file paths
    index_dict_file_path =  '/raid/ankit/lstm/we_index_dict{}.pkl'.format(masking_value)
    embedding_weights_file_path = '/raid/ankit/lstm/we_embedding_weights{}.pkl'.format(masking_value)
    word_vectors_file = "/raid/ankit/vectors/vectors_wholecorpus100.txt"

    if not use_pickled:
        index_dict = {}
        # adding 1 to account for 0th index (for masking) [Number of word:vector pairs is 7115783]
        n_symbols = 7556273 # TODO: Figure out a way to remove this hard-coded value!
        embedding_weights = np.zeros((n_symbols + 1, embedding_dim))
        # Path to word vectors file

        if masking:
            i = 0   # Leaves the 0-index free of any data.
        else:
            i = -1  # Stores the embedding weights from the zero'th index itself.

        words_set = set()
        with open(word_vectors_file) as fopen:
            for line in fopen:
                i += 1
                try:
                    components = line.strip().split()
                    if not len(components) < embedding_dim:

                        if i % 1000000 == 0:
                            print("Words added to embedding matrix ... {}".format(i))

                        word = components[0]
                        words_set.add(word)
                        vec = np.asarray([float(x) for x in components[1:embedding_dim + 1]])
                        index_dict[word] = i
                        embedding_weights[i, :] = vec
                except Exception as e:
                    print("Exception Encountered: ".format(e))
        print("Word Embeddings added. Current value of i : {}".format(i))
        # Adding the word vectors from the input datasets which are not in the word vector file.
        # Word Vectors are drawn at random from a uniform distribution(-0.25, 0.25)
        for word_k in vocab_list:
            if not word_k in words_set and not word_k in index_dict:
                i += 1
                index_dict[word_k] = i
                embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)
        print("Added Random Vectors for the unseen words in the corpus. Current value of i: {}".format(i))
        print("Dumping embedding weights and index_dict to disk as pickled files ....")
        joblib.dump(index_dict, index_dict_file_path)
        joblib.dump(embedding_weights, embedding_weights_file_path)
        print('Finished: Dumping index_dict and embedding_weights to disk.')
        return index_dict, embedding_weights
    else:
        print('Loading Word Embeddings: index_dict and embeddings weights from disk ... ')
        index_dict = joblib.load(index_dict_file_path)
        embedding_weights = joblib.load(embedding_weights_file_path)
        print("Word Embedding pickled files loaded into memory!")
        return index_dict, embedding_weights


def load_word_embeddings_compact(embedding_dim, vocab_list, masking=False, use_pickled=True):

    print("Loading Word Embeddings into memory ... ")
    if masking:
        masking_value = ""  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
    else:
        masking_value = "_non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"

    # Dataset sources file paths
    index_dict_file_path =  '/raid/ankit/lstm/we_index_dict_compact{}.pkl'.format(masking_value)
    embedding_weights_file_path = '/raid/ankit/lstm/we_embedding_weights_compact{}.pkl'.format(masking_value)
    word_vectors_file = "/raid/ankit/vectors/vectors_wholecorpus100.txt"

    if not use_pickled:
        index_dict = {}

        # Path to word vectors file

        if masking:
            i = 0   # Leaves the 0-index free of any data.
        else:
            i = -1  # Stores the embedding weights from the zero'th index itself.

        word_vector_dict = {}

        j=0
        with open(word_vectors_file) as fopen:
            for line in fopen:
                j += 1
                try:
                    components = line.strip().split()
                    if not len(components) < embedding_dim:
                        if j % 1000000 == 0:
                            print("Words added to embedding matrix ... {}".format(j))
                        word = components[0]
                        vec = np.asarray([float(x) for x in components[1:embedding_dim + 1]])
                        word_vector_dict[word] = vec

                except Exception as e:
                    print("Exception Encountered: ".format(e))
        print("Word Embeddings added to word_vector_dict. Current value of i : {}".format(i))
        # Adding the word vectors from the input datasets which are not in the word vector file.
        # Word Vectors are drawn at random from a uniform distribution(-0.25, 0.25)
        # adding 1 to account for 0th index (for masking) [Number of word:vector pairs is 7115783]
        n_symbols = len(vocab_list)  # TODO: Figure out a way to remove this hard-coded value!
        embedding_weights = np.zeros((n_symbols + 1, embedding_dim))

        for word_k in vocab_list:
            i += 1
            index_dict[word_k] = i
            if word_k in word_vector_dict:
                embedding_weights[i, :] = word_vector_dict[word_k]
            else:
                embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

        print("Added Random Vectors for the unseen words in the corpus. Current value of i: {}".format(i))
        print("Dumping embedding weights and index_dict to disk as pickled files ....")
        joblib.dump(index_dict, index_dict_file_path)
        joblib.dump(embedding_weights, embedding_weights_file_path)
        print('Finished: Dumping index_dict and embedding_weights to disk.')
        return index_dict, embedding_weights
    else:
        print('Loading Word Embeddings: index_dict and embeddings weights from disk ... ')
        index_dict = joblib.load(index_dict_file_path)
        embedding_weights = joblib.load(embedding_weights_file_path)
        print("Word Embedding pickled files loaded into memory!")
        return index_dict, embedding_weights


def load_data(embedding_dim, embedding_weights_masking, load_embeddings_pickled=True):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    # sentences, labels = load_data_and_labels()
    query_text, similar_text, nonsimilar_text1, nonsimilar_text2, nonsimilar_text3, \
        data_output_labels = load_data_and_labels()

    query_text_padded = pad_sentences(query_text)
    similar_text_padded = pad_sentences(similar_text)
    nonsimilar_text1_padded = pad_sentences(nonsimilar_text1)
    nonsimilar_text2_padded = pad_sentences(nonsimilar_text2)
    nonsimilar_text3_padded = pad_sentences(nonsimilar_text3)

    # Get a list of all words from various datasets.
    vocabulary_query_inv = build_vocab(query_text_padded)
    vocabulary_similar_inv = build_vocab(similar_text_padded)
    vocabulary_nonsimilar1_inv = build_vocab(nonsimilar_text1_padded)
    vocabulary_nonsimilar2_inv = build_vocab(nonsimilar_text2_padded)
    vocabulary_nonsimilar3_inv = build_vocab(nonsimilar_text3_padded)

    vocab_list  = list(set(vocabulary_query_inv + vocabulary_similar_inv + vocabulary_nonsimilar1_inv + vocabulary_nonsimilar2_inv + vocabulary_nonsimilar3_inv))

    #index_dict, embedding_weights = load_word_embeddings(embedding_dim, vocab_list, masking=embedding_weights_masking, use_pickled =load_embeddings_pickled)
    index_dict, embedding_weights = load_word_embeddings_compact(embedding_dim, vocab_list, masking=embedding_weights_masking, use_pickled =load_embeddings_pickled)

    print('Build Input Data for queries and documents (similar and non-similar)... ')
    x_query = build_input_data(query_text_padded, index_dict)
    x_similar = build_input_data(similar_text_padded, index_dict)
    x_nonsimilar1 = build_input_data(nonsimilar_text1_padded,  index_dict)
    x_nonsimilar2 = build_input_data(nonsimilar_text2_padded, index_dict)
    x_nonsimilar3 = build_input_data(nonsimilar_text3_padded, index_dict)

    final_out = [
        ([x_query, vocabulary_query_inv],
        [x_similar, vocabulary_similar_inv],
        [x_nonsimilar1, vocabulary_nonsimilar1_inv],
        [x_nonsimilar2,  vocabulary_nonsimilar2_inv],
        [x_nonsimilar3,  vocabulary_nonsimilar3_inv]), data_output_labels, index_dict
    ]

    print "Data Generated.. Now Dumping to disk....."
    joblib.dump(final_out, "/raid/ankit/lstm/lstm_input_data_compact.pkl")
    print "Data Dumped to disk!"
    return final_out, embedding_weights


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
