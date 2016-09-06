#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import urllib2
import sys
import math
import gzip
import json
import shutil
from datetime import datetime
from itertools import izip_longest
from log import setup_log
setup_log.setup_logging()
import logging
logger = logging.getLogger(__name__)

class GeneralUtils(object):
    def __init__(self):
        pass

    def get_absolute_path_script(self):
        """
        Returns the absolute path of the script on the file system.
        """
        return str(os.path.realpath(__file__))

    def get_absolute_path_script_dir(self):
        """
        Returns the absolute path of the directory where the script lies in the filesystem.
        """
        return str(os.path.dirname(os.path.realpath(__file__)))

    def get_absolute_path_head_dir(self, head_dir_name):
        """
        Returns the path inclusive of "head_dir_name". Used to identify the base project directory name in config.py
        :param head_dir_name: str
        :rtype: str
        """
        k = self.get_absolute_path_script_dir()
        rr = k.split("/")
        flag = True
        oo = []
        for i in rr:
            if flag == True:
                if i == head_dir_name:
                    oo.append(i)
                    flag = False
                else:
                    oo.append(i)

        return "/".join(oo)

    def create_dir(self, dir_name):
        """ Creates a new directory recursively if not already present. """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def download_http_data(self, url, output_file_path):
        """ Download HTTP content from the URL to the location specified by the output_file_path"""
        u = urllib2.urlopen(url)
        f = open(output_file_path, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        logger.info("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            p = float(file_size_dl) / file_size
            # status = r"{0}  [{1:.2%}]".format(file_size_dl, p)
            # status = status + chr(8) * (len(status) + 1)
            # sys.stdout.write(status)

        f.close()

    def get_gzcat_s3_dir(self, s3_dir, combined_output_file):
        """ Gzcat *.gz files in s3_dir directory into one single file specified by the combined_output_file  """
        os.system("zcat {}*.gz > {}".format(self.norm_dir_string(s3_dir), combined_output_file))
        logger.info("Finished zcat extraction into: {}".format(combined_output_file.split("/")[-1]))

    def get_gzcat_file(self, input_file, output_file):
        """ Gzcat inputfile.gz file into output_file location """
        if input_file.endswith(".gz"):
            os.system("unpigz --stdout --keep {} > {}".format(input_file, output_file))
            logger.info("Finished zcat extraction into: {}".format(output_file.split("/")[-1]))

    def get_list_file_in_dir(self, dir_path, prefix=""):
        """
        Returns the names of all the files in a given derectory.
        If the prefix is passed then only those names will be returned in the list which have the given prefix.
        """
        dir_file_list = []
        for root, dirs, files in os.walk(dir_path):
            path = root.split('/')
            for file in files:
                if not prefix == "":
                    if file.startswith(prefix):
                        dir_file_list.append(str(file))
                else:
                    dir_file_list.append(str(file))
        return dir_file_list

    def normalize_redis_vector(self, vector_tmp):
        """ Normalized the Redis Vector from a string to a list of floats."""
        vector_norm = map(float, vector_tmp.split())
        return vector_norm

    def grouper(self, n, iterable, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return izip_longest(fillvalue=fillvalue, *args)

    def group_list_concurrency(self, inputdata_list, num_processes):
        """
        Groups a list into a list of n elements per item
        :param inputdata_list: list
        :param num_processes: number of elements in an item in the output list

        Example:
        ('abcdefg', 3) -->
        ('a','b','c'), ('d','e','f'), ('g')
        """
        parameter_list_final = []
        for i in range(0, int(math.ceil(len(inputdata_list) / float(num_processes)))):
            parameter_list_final.append(inputdata_list[i * num_processes:(i + 1) * num_processes])
        return parameter_list_final

    def get_current_date_time(self):
        """
        Returns the date and time as string in given format:  %Y-%m-%dT%H-%M-%S
        Where, Y: year, m: month, d: day, H: Hour, M: Minute amd S: second
        """
        t = datetime.today()
        date_time_str = t.strftime('%Y-%m-%dT%H-%M-%S')
        return date_time_str

    def convert_line_unicode(self, line):
        """ UTF-8 encodes a string (if not already) and returns the encoded string"""
        try:
            if not isinstance(line, unicode):
                line = line.encode("utf-8")
        except:
            pass
        return line

    def norm_dir_string(self, dir_path):
        """ Normalize the directory path string, returns a string with a '/' at the end. """
        if not dir_path.endswith("/"):
            return str(dir_path.strip()) + "/"
        return dir_path

    def gzip_file(self, input_file_path, output_file_path):
        """ Gzip compress a file at the input_file_path to the output_file_path """
        os.system("cat {} | pigz --fast > {}".format(input_file_path, output_file_path))


    def split_combined_file(self, combined_file, num_splits, split_file_path_template, num_lines=0):
        """
        Splits a large combined file into equal parts according to num_splits parameter supplied.
        """
        logger.info("Splitting {} into equal parts ... ".format(os.path.basename(combined_file)))
        if num_lines == 0:
            num_lines = sum(1 for line in open(combined_file))
        split_size = int(num_lines / num_splits) + 1

        count = 0
        cc = True
        i = 0
        fw = open(split_file_path_template.format(i), "w")
        with open(combined_file) as f:
            for line in f:
                if cc == True:
                    logger.info("Currently writing - split file: {}".format(split_file_path_template.split("/")[-1].format(i)))
                    cc = False
                if count == split_size:
                    line = self.convert_line_unicode(line)
                    fw.write(line)
                    logger.info("File:{} -- Done!".format(split_file_path_template.split("/")[-1].format(i)))
                    cc = True
                    count = 0
                    i += 1
                    fw.close()
                    fw = open(split_file_path_template.format(i), "w")
                line = self.convert_line_unicode(line)
                fw.write(line)
                count += 1

        logger.info("Finished splitting: {}".format(os.path.basename(combined_file)))


    def remove_dir_contents(self, dir_path, remove_sub_dir=False):
        """ Remove the directory contents and preserving the empty directory"""
        logger.info("Removing contents of the directory")
        for the_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and remove_sub_dir==True:
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error("Remove directory contents error: {}".format(e))

    def combine_multiple_files(self, input_dir, input_file_list, combined_output_file, delete_split_files=False):
        """ Combine multiple text files into one file"""
        logger.info("Starting combining the files ... ")
        with open(combined_output_file, 'wb') as outfile:
            for filename in input_file_list:
                if input_dir+filename == combined_output_file:
                    # don't want to copy the output into the output
                    continue
                with open(input_dir+filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, outfile)
        logger.info("Finished combining file(s) to: {}".format(os.path.basename(combined_output_file)))
        # Deleting Split files after combined to one single file
        if delete_split_files:
            logger.info("Starting: Deletion of input split files.")
            for filename in input_file_list:
                if input_dir+filename == combined_output_file:
                    # don't want to copy the output into the output
                    continue
                os.remove(input_dir+filename)
        logger.info("Finished: Deleting the number of split files.")


