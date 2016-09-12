import numpy as np

import ujson as json
import joblib
from config import Configuration
from utils.gen_utils import GeneralUtils
from utils.data_utils import DataUtils

# Class Object Initialization.
conf = Configuration()
genutil = GeneralUtils()
du = DataUtils()


def load_word_embeddings_compact(embedding_dim, vocab_list, masking=False, use_pickled=True):

    print("Loading Word Embeddings into memory ... ")
    if masking:
        masking_value = "_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
    else:
        masking_value = "_non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"

    # Dataset sources file paths
    embedding_weights_file_path = conf.embedding_weights_file_tpl.format(masking_value)


    if not use_pickled:
        # Path to word vectors file

        if masking:
            i = 0   # Leaves the 0-index free of any data.
        else:
            i = -1  # Stores the embedding weights from the zero'th index itself.

        word_vector_dict = {}

        j=0
        with open(conf.word_vectors_file) as fopen:
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

            if word_k in word_vector_dict:
                embedding_weights[i, :] = word_vector_dict[word_k]
            else:
                embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

        print("Added Random Vectors for the unseen words in the corpus. Current value of i: {}".format(i))
        if conf.create_data_dump:
            print("Dumping embedding weights and index_dict to disk as pickled files ....")
            joblib.dump(embedding_weights, embedding_weights_file_path)
            print('Finished: Dumping index_dict and embedding_weights to disk.')
        return embedding_weights
    else:
        print('Loading Word Embeddings: index_dict and embeddings weights from disk ... ')
        embedding_weights = joblib.load(embedding_weights_file_path)
        print("Word Embedding pickled files loaded into memory!")
        return embedding_weights


def generate_vocabulary_set(model_training_data_file, masking=False):
    # Load data from files
    print "Generating Vocabulary set from Model Training Data: {}".format(model_training_data_file)

    q = []
    sim = []
    ns = []
    vocab_index_dict = {}

    # Data-Set Line Format: {'q': query, 'doc_corr': correct_url_doc, 'doc_incorr': incorrect_doc_list}
    less_doc_cnt = 0

    for model_training_data_file in conf.input_file_list:
        with open(model_training_data_file) as fo:
            for line in fo:
                data = json.loads(line)
                if len(data['doc_incorr']) == conf.num_negative_examples:
                    q.append(data['q'])
                    sim.append(data['doc_corr'])
                    ns.append(data['doc_incorr'])
                else:
                    less_doc_cnt += 1

        tmp_list = [q, sim, ns]
        vocab_set = set()

        # VOCAB and Padded List Generation
        for x in tmp_list:
            if len(x) == 1:
                x_word_split = du.get_text_word_splits(x)
                x_padded_text = du.pad_sentences(x_word_split)
                x_vocab = du.build_vocab(x_padded_text)
                for i in x_vocab:
                    if not i in vocab_set:
                        vocab_set.add(i)
            else:
                for i in xrange(0, len(x)):
                    x_word_split = du.get_text_word_splits(x[i])
                    x_padded_text = du.pad_sentences(x_word_split)
                    x_vocab = du.build_vocab(x_padded_text)
                    for i in x_vocab:
                        if not i in vocab_set:
                            vocab_set.add(i)

    if masking:
        i = 0
    else:
        i = -1

    for word in vocab_set:
        i += 1
        vocab_index_dict[word] = i

    if conf.create_data_dump:
        print "Dumping Vocabulary Set and Index - dict to Disk!"
        joblib.dump(vocab_set, conf.vocab_set_file)
        joblib.dump(vocab_index_dict, conf.vocab_index_file)
    return vocab_set, vocab_index_dict



def load_data_generator(embedding_dim, embedding_weights_masking, load_embeddings_pickled=True, load_vocab_pickled=True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    if load_vocab_pickled:
        vocab_index_dict = joblib.load(conf.vocab_index_file)
    else:
        vocab_set, vocab_index_dict = generate_vocabulary_set(conf.model_training_data, masking=False)

    print "Loading Model Training Data: {}".format(conf.model_training_data)

    # Data-Set Line Format: {'q': query, 'doc_corr': correct_url_doc, 'doc_incorr': incorrect_doc_list}
    less_doc_cnt = 0
    with open(conf.model_training_data) as fo:
        for line in fo:
            line_data = []
            data = json.loads(line)
            if len(data['doc_incorr']) == conf.num_negative_examples:
                query = [data['q']]
                correct_doc = [data['doc_corr']]
                incorr_doc = data['doc_incorr']
                input_data_list = [query, correct_doc, incorr_doc]
                res = []
                # Build Input Data
                for x in input_data_list:
                    if len(x) == 1:
                        x_input = du.build_input_data(du.pad_sentences(du.get_text_word_splits(x)), vocab_index_dict, return_array=True)
                        res.append(x_input)
                    else:
                        for i in xrange(0, len(x)):
                            x_input = du.build_input_data(du.pad_sentences(du.get_text_word_splits(x)),
                                                          vocab_index_dict, return_array=True)
                            res.append(x_input)
                # Generate labels
                yield res, np.ones(1)

            else:
                less_doc_cnt +=1
    print "Number of skipped data points: Incorrect Documents in Training Data (< 3): {}".format(less_doc_cnt)


def get_vocab_index_embedding_weights(embedding_dim, embedding_weights_masking, load_embeddings_pickled):
    vocab_set = generate_vocabulary_set(conf.model_training_data)
    vocab_index_dict, embedding_weights = load_word_embeddings_compact(embedding_dim, list(vocab_set),
                                                                       masking=embedding_weights_masking,
                                                                       use_pickled=load_embeddings_pickled)
    return embedding_weights, vocab_index_dict


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
