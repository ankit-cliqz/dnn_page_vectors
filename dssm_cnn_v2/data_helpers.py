import joblib
import numpy as np
import ujson as json
from config import Configuration
from utils.gen_utils import GeneralUtils
from utils.data_utils import DataUtils
from itertools import islice, chain

class DataHelpers(object):
    def __init__(self):
        # Class Object Initialization.
        self.conf = Configuration()
        self.genutil = GeneralUtils()
        self.du = DataUtils()


    def load_word_embeddings_compact(self, embedding_dim, vocab_set, masking=False, use_pickled=True):

        vocab_list = list(vocab_set)

        if masking:
            masking_value = "masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
            start_index = 1  # Leaves the 0-index free of any data.
        else:
            masking_value = "non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
            start_index = 0  # Stores the embedding weights from the zero'th index itself.

        # Dataset sources file paths
        embedding_weights_file_path = self.conf.embedding_weights_file_tpl.format(masking_value)


        if not use_pickled:
            print("Loading Word Embeddings into memory ... ")
            word_vector_dict = {}
            j=0
            with open(self.conf.word_vectors_file, "r") as fopen:
                for line in fopen:
                    j += 1
                    try:
                        components = line.strip().split()
                        if not len(components) < embedding_dim:
                            if j % 1000000 == 0:
                                print("Parsing word vector file ... {}".format(j))
                            word = components[0]
                            if word in vocab_set:
                                vec = np.asarray([float(x) for x in components[1:embedding_dim + 1]])
                                word_vector_dict[word] = vec

                    except Exception as e:
                        print("Exception Encountered: ".format(e))
            print("Word Embeddings added to word_vector_dict.")
            # Adding the word vectors from the input datasets which are not in the word vector file.
            # Word Vectors are drawn at random from a uniform distribution(-0.25, 0.25)
            # adding 1 to account for 0th index (for masking) [Number of word:vector pairs is 7115783]
            n_symbols = len(vocab_list)
            embedding_weights = np.zeros((n_symbols + 1, embedding_dim))

            for i, word_k in enumerate(vocab_list, start=start_index):
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

        # Adding a padding word, unknown word and space
        vocab_set.add('<PAD/>')
        vocab_set.add('<UNK/>')
        vocab_set.add(' ')

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
                        x_vocab = self.du.build_vocab(self.du.get_text_feature_splits(s_list, mode=self.conf.feature_level))
                        for i in x_vocab:
                            if not i in vocab_set:
                                # print "Vocab_Entity: {}".format(i.encode('utf-8'))
                                vocab_set.add(i)
                    else:
                        less_doc_cnt += 1

        if masking:
            i = 0
            masking_value = "masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
        else:
            i = -1
            masking_value = "non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"

        for word in vocab_set:
            i += 1
            vocab_index_dict[word] = i

        if self.conf.create_data_dump:
            print "Dumping Vocabulary Set and Index - dict to Disk!"
            joblib.dump(vocab_set, self.conf.vocab_set_file.format(masking_value))
            joblib.dump(vocab_index_dict, self.conf.vocab_index_file.format(masking_value))
        return vocab_set, vocab_index_dict



    def load_data_generator(self, vocab_index_dict, mode=None, batch_size=128, nb_epochs=1):
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


        # print "\nLoading Model Training Data: {}\n".format(input_dataset_file)
        # Data-Set Line Format: {'q': query, 'doc_corr': correct_url_doc, 'doc_incorr': incorrect_doc_list}

        for epoch in range(0, nb_epochs+2):
            less_doc_cnt = 0
            with open(input_dataset_file, 'r') as fin:
                while True:
                    batch_rows = list(islice(fin, batch_size))

                    if not batch_rows:
                        break

                    batch_query_data = np.empty( shape=(0, 0)  , dtype=np.int32)
                    batch_pos_query_data = np.empty( shape=(0, 0)  , dtype=np.int32)
                    batch_neg_query_data = [np.empty( shape=(0, 0)  , dtype=np.int32) for _ in range(0, self.conf.num_negative_examples)]

                    for line in batch_rows:
                        data = json.loads(line)
                        if len(data['doc_incorr']) == self.conf.num_negative_examples:
                            input_data_list = [[data['q']], [data['doc_corr']], data['doc_incorr']]
                            # Build Input Data
                            for n, x in enumerate(input_data_list):
                                if n==0:
                                    for i in xrange(0, len(x)):
                                        x_array = self.du.build_input_data(self.du.pad_sentences(self.du.get_text_feature_splits(x[i], cutoff=self.conf.query_length, mode=self.conf.feature_level), self.conf.query_length),
                                                                           vocab_index_dict, return_array=True)
                                        if batch_query_data.shape[0]==0:
                                            batch_query_data = x_array
                                        else:
                                            batch_query_data = np.vstack((batch_query_data, x_array))


                                elif n==1:
                                    for i in xrange(0, len(x)):
                                        x_array = self.du.build_input_data(self.du.pad_sentences(self.du.get_text_feature_splits(x[i], cutoff=self.conf.document_length, mode=self.conf.feature_level), self.conf.document_length),
                                                                           vocab_index_dict, return_array=True)
                                        if batch_pos_query_data.shape[0]==0:
                                            batch_pos_query_data = x_array
                                        else:
                                            batch_pos_query_data =np.vstack((batch_pos_query_data, x_array))
                                elif n==2:
                                    for i in xrange(0, len(x)):
                                        x_array = self.du.build_input_data(self.du.pad_sentences(self.du.get_text_feature_splits(x[i], cutoff=self.conf.document_length, mode=self.conf.feature_level), self.conf.document_length),
                                                                           vocab_index_dict, return_array=True)
                                        if batch_neg_query_data[i].shape[0]==0:
                                            batch_neg_query_data[i] = x_array
                                        else:
                                            batch_neg_query_data[i] = np.vstack((batch_neg_query_data[i], x_array))
                        else:
                            less_doc_cnt += 1

                    batch_y_data = np.ones(len(batch_query_data))
                    yield [batch_query_data, batch_pos_query_data] + batch_neg_query_data, batch_y_data

            #print "Number of skipped data points: Incorrect Documents in Training Data (< 3): {}".format(less_doc_cnt)


    def get_vocab_index_embedding_weights(self, embedding_dim, embedding_weights_masking, load_embeddings_pickled=False, load_vocab_pickled=False):
        embedding_weights = []
        if embedding_weights_masking==True:
            masking_value = "masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"
        else:
            masking_value = "non_masked"  # For masked embedding weights leave it blank "", else for masked use "_non_masked"

        # Load data from files
        if load_vocab_pickled:
            vocab_index_dict = joblib.load(self.conf.vocab_index_file.format(masking_value))
            vocab_set = joblib.load(self.conf.vocab_set_file.format(masking_value))

        else:
            vocab_set, vocab_index_dict = self.generate_vocabulary_set(masking=embedding_weights_masking)


        if self.conf.feature_level == "word":
            embedding_weights = self.load_word_embeddings_compact(embedding_dim, vocab_set,
                                                                           masking=embedding_weights_masking,
                                                                           use_pickled=load_embeddings_pickled)
        return embedding_weights, vocab_index_dict

