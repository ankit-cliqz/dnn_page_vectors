#!/usr/bin/python
# -*- coding: UTF-8 -*-
import random
from config import Configuration
import os
from utils.gen_utils import GeneralUtils

class SetupExperiment(object):

    def __init__(self):
        random.seed(1337)
        self.conf = Configuration()
        self.genutil = GeneralUtils()



    def split_dataset_file(self, input_dataset_file):
        if not os.path.exists(self.conf.model_training_data) and not os.path.exists(self.conf.model_validation_data):
            print "Splitting input dataset file into Training and Validation sets .."
            fw_train_data = open(self.conf.model_training_data, "w")
            fw_validation_data = open(self.conf.model_validation_data, "w")

            num_lines = sum(1 for line in open(input_dataset_file))

            num_train_batch = int(num_lines * (1 - self.conf.train_validation_split)) + 1
            train_batch = set(random.sample(range(0,num_lines+1), num_train_batch))
            validation_batch = set([x for x in  range(0,num_lines+1) if x not in train_batch])
            cnt = 0
            with open(input_dataset_file, "r") as fo:
                for line in fo:
                    if cnt in train_batch:
                        fw_train_data.write(line.strip()+"\n")
                    elif cnt in validation_batch:
                        fw_validation_data.write(line.strip()+"\n")
                    cnt+=1
            fw_train_data.close()
            fw_validation_data.close()
            print "Splitting of input dataset ... Finished!"
        else:
            print "Input data already split into training and validation splits!"

    def create_workspace(self):
        print "Creating Work space ... "
        self.genutil.create_dir(self.conf.data_path)
        self.genutil.create_dir(self.conf.data_dir)
        self.genutil.create_dir(self.conf.trained_model_dir)
        self.genutil.create_dir(self.conf.pickle_files_dir)

    def download_dataset(self):
        os.system('aws s3 cp {} {}'.format(self.conf.input_dataset_s3_path , self.conf.input_dataset))


if __name__=="__main__":
    exp = SetupExperiment()
    print "Setting up Experiment: {}".format(exp.conf.experiment_name)
    exp.create_workspace()
    exp.download_dataset()
    exp.split_dataset_file(exp.conf.input_dataset)








