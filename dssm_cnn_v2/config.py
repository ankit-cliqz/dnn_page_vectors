#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from utils.gen_utils import GeneralUtils
import ujson as json


class Configuration(object):
    """
    Configuration File for Document Vectors Project
    """

    def __init__(self):

        genutil = GeneralUtils()

        self.experiment_root_directory = "/ebs/project_data"
        self.reuse_experiment_timestamp = True
        self.experiment_name = "dssm_cnn_v2"

        tmp_timestamp_file = os.path.join(self.experiment_root_directory, "_TIMESTAMP")

        if self.reuse_experiment_timestamp:
            timestamp = "2016-09-16T23-04-38"
        else:
            if os.path.exists(tmp_timestamp_file):
                with open(tmp_timestamp_file) as json_data:
                    d = json.load(json_data)
                    timestamp = d["project_timestamp"]

            else:
                timestamp = genutil.get_current_date_time()
                fw = open(tmp_timestamp_file, "w")
                fw.write(json.dumps({ "project_timestamp" : timestamp}))
                fw.close()

        # Feature Level of Text Input
        self.feature_level = "char"  # ["word", "ngram", "char"]
        self.default_ngram = 3

        self.data_path = os.path.join(self.experiment_root_directory, self.experiment_name, timestamp, self.feature_level)

        self.data_dir = os.path.join(self.data_path, "data")

        self.trained_model_dir = os.path.join(self.data_path, "model")

        self.pickle_files_dir = os.path.join(self.data_path, "pickled_files")

        self.vectors_directory = os.path.join(self.experiment_root_directory, "vectors")

        self.word_vectors = "fast" # Word Vectors Computation Algorithm

        if self.word_vectors == "word2vec":
            self.word_vectors_file = os.path.join(self.vectors_directory, "vectors_wholecorpus100.txt")
        elif self.word_vectors == "fast":
            self.word_vectors_file = os.path.join(self.vectors_directory, "fast_model_ns.vec")

        # Remote s3 location, where the dataset is located.
        self.input_dataset_s3_path = "s3://ankit-test/ebs_backup/lstm/model_training_data.txt"
        self.word2vec_wordvector_s3_path = "s3://ankit-test/vectors_final/vectors_wholecorpus100.txt"
        self.fast_wordvector_s3_path = "s3://ankit-test/fast_model/full_model.vec"

        self.input_dataset =os.path.join(self.data_dir, "input_dataset_new.txt")

        self.model_training_data =os.path.join(self.data_dir, "model_training_data_new.txt")

        self.model_validation_data =os.path.join(self.data_dir, "model_validation_data_new.txt")

        self.input_file_list = [self.model_training_data, self.model_validation_data]

        self.create_data_dump = True

        self.vocab_set_file = os.path.join(self.pickle_files_dir, "vocab_set_{}.pkl")
        self.vocab_index_file = os.path.join(self.pickle_files_dir,  "vocab_index_dict_{}.pkl")
        self.embedding_weights_file_tpl = os.path.join(self.pickle_files_dir, 'we_embedding_weights_compact_{}.pkl')
        self.num_negative_examples = 3

        self.train_validation_split = 0.2 # How much Percentage of the original data needs to be considered as validation split

        if self.feature_level == 'word':
            # Heuristical Max-Cutoff of Length of Document
            self.query_length = 20
            self.document_length = 975
        elif self.feature_level == 'char':
            # Heuristical Max-Cutoff of Length of Document
            self.query_length = 250
            self.document_length = 5000
        if self.feature_level == 'ngram':
            # Heuristical Max-Cutoff of Length of Document
            self.query_length = 45
            self.document_length = 2000





