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

        self.experiment_root_directory = "/raid/ankit"
        self.reuse_experiment_timestamp = False
        self.experiment_name = "dssm_cnn_v2"

        tmp_timestamp_file = os.path.join(self.experiment_root_directory, "/_TIMESTAMP")

        if self.reuse_experiment_timestamp:
            timestamp = ""
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

        self.data_path = os.path.join(self.experiment_root_directory, self.experiment_name, timestamp)
        genutil.create_dir(self.data_path)

        self.data_dir = os.path.join(self.data_path, "data")
        genutil.create_dir(self.data_dir)

        self.trained_model_dir = os.path.join(self.data_path, "model")
        genutil.create_dir(self.trained_model_dir)

        self.pickle_files_dir = os.path.join(self.data_path, "pickled_files")
        genutil.create_dir(self.pickle_files_dir)

        self.word_vectors = "fast" #fast

        if self.word_vectors == "word2vec":
            self.word_vectors_directory = os.path.join(self.experiment_root_directory, "vectors", "vectors_wholecorpus100.txt")
        elif self.word_vectors == "fast":
            self.word_vectors_directory = os.path.join(self.experiment_root_directory, "vectors", "fast_model_ns.vec")

        self.model_training_data =os.path.join(self.data_dir, "model_training_data.txt")