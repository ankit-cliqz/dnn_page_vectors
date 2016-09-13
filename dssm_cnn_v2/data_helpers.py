import numpy as np

import ujson as json
import joblib
from config import Configuration
from utils.gen_utils import GeneralUtils
from utils.data_utils import DataUtils




class DataHelpers(object):
    def __init__(self):
        # Class Object Initialization.
        self.conf = Configuration()
        self.genutil = GeneralUtils()
        self.du = DataUtils()


    def load_word_embeddings_compact(self, embedding_dim, vocab_list, masking=False, use_pickled=True):

        print("Loading Word Embeddings into memory ... ")
        if masking:
            masking_value = "_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
        else:
            masking_value = "_non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"

        # Dataset sources file paths
        embedding_weights_file_path = self.conf.embedding_weights_file_tpl.format(masking_value)


        if not use_pickled:
            # Path to word vectors file

            if masking:
                i = 0   # Leaves the 0-index free of any data.
            else:
                i = -1  # Stores the embedding weights from the zero'th index itself.

            word_vector_dict = {}

            j=0
            with open(self.conf.word_vectors_file) as fopen:
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
            n_symbols = len(vocab_list)
            embedding_weights = np.zeros((n_symbols + 1, embedding_dim))

            for word_k in vocab_list:

                if word_k in word_vector_dict:
                    embedding_weights[i, :] = word_vector_dict[word_k]
                else:
                    embedding_weights[i, :] = np.random.uniform(-0.25, 0.25, embedding_dim)

            print("Added Random Vectors for the unseen words in the corpus. Current value of i: {}".format(i))
            if self.conf.create_data_dump:
                print("Dumping embedding weights and index_dict to disk as pickled files ....")
                joblib.dump(embedding_weights, embedding_weights_file_path)
                print('Finished: Dumping index_dict and embedding_weights to disk.')
            return embedding_weights
        else:
            print('Loading Word Embeddings: index_dict and embeddings weights from disk ... ')
            embedding_weights = joblib.load(embedding_weights_file_path)
            print("Word Embedding pickled files loaded into memory!")
            return embedding_weights


    def generate_vocabulary_set(self, masking=False):
        # Load data from files
        print "Generating Vocabulary set from Input Data file (s): {}".format(self.conf.input_file_list)

        vocab_index_dict = {}
        vocab_set = set()

        # Adding a padding word and unknown word
        vocab_set.add('<PAD/>')
        vocab_set.add('<UNK/>')

        # Data-Set Line Format: {'q': query, 'doc_corr': correct_url_doc, 'doc_incorr': incorrect_doc_list}
        less_doc_cnt = 0

        for model_training_data_file in self.conf.input_file_list:
            with open(model_training_data_file) as fo:
                for line in fo:
                    data = json.loads(line)
                    if len(data['doc_incorr']) == self.conf.num_negative_examples:
                        s_list = []
                        s_list.append(data['q'])
                        s_list.append(data['doc_corr'])
                        s_list +=  data['doc_incorr']
                        x_vocab = self.du.build_vocab(self.du.get_text_word_splits(s_list))
                        for i in x_vocab:
                            if not i in vocab_set:
                                vocab_set.add(i)
                    else:
                        less_doc_cnt += 1

        if masking:
            i = 0
        else:
            i = -1

        for word in vocab_set:
            i += 1
            vocab_index_dict[word] = i

        if self.conf.create_data_dump:
            print "Dumping Vocabulary Set and Index - dict to Disk!"
            joblib.dump(vocab_set, self.conf.vocab_set_file)
            joblib.dump(vocab_index_dict, self.conf.vocab_index_file)
        return vocab_set, vocab_index_dict



    def load_data_generator(self, embedding_dim, vocab_index_dict, embedding_weights_masking, load_embeddings_pickled=True, load_vocab_pickled=True, mode=None, batch_size=128, nb_epochs=100, negs=5):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        if mode==None:
            raise Exception("Please provide mode as either 'training' or 'validation'")


        input_dataset_file= ""
        if mode=="training":
            input_dataset_file = self.conf.model_training_data
        elif mode=="validation":
            input_dataset_file = self.conf.model_validation_data


        print "Loading Model Training Data: {}".format(self.conf.model_training_data)
        # Data-Set Line Format: {'q': query, 'doc_corr': correct_url_doc, 'doc_incorr': incorrect_doc_list}
        less_doc_cnt = 0
        with open(input_dataset_file) as fo:
            for line in fo:
                line_data = []
                data = json.loads(line)
                if len(data['doc_incorr']) == self.conf.num_negative_examples:
                    query = [data['q']]
                    correct_doc = [data['doc_corr']]
                    incorr_doc = data['doc_incorr']
                    input_data_list = [query, correct_doc, incorr_doc]
                    res = []
                    # Build Input Data
                    for x in input_data_list:
                        for i in xrange(0, len(x)):
                            x_input = self.du.build_input_data(self.du.pad_sentences(self.du.get_text_word_splits(x[i])),
                                                          vocab_index_dict, return_array=True)
                            res.append(x_input)
                    # Generate labels
                    yield res, np.ones(1)

                else:
                    less_doc_cnt +=1
        print "Number of skipped data points: Incorrect Documents in Training Data (< 3): {}".format(less_doc_cnt)


    def get_vocab_index_embedding_weights(self, embedding_dim, embedding_weights_masking, load_embeddings_pickled):
        vocab_set = self.generate_vocabulary_set(self.conf.model_training_data)
        vocab_index_dict, embedding_weights = self.load_word_embeddings_compact(embedding_dim, list(vocab_set),
                                                                           masking=embedding_weights_masking,
                                                                           use_pickled=load_embeddings_pickled)
        return embedding_weights, vocab_index_dict


    def batch_iter(self, data, batch_size, num_epochs):
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
