#!/usr/bin/python
# -*- coding: UTF-8 -*-

#!/usr/local/bin/python
# -*- coding: utf-8 -*-

__author__ = 'ankit'

from itertools import izip_longest
from HTMLParser import HTMLParser
import re
import math
class MLStripper(HTMLParser):
    def __init__(self):
        # super(MLStripper, self).__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


class ProcessHTMLContent(object):
    """
    Custom class to parse HTML tags and other content.
    Uses the standard HTML parser
    """

    def strip_tags(self, html):
        s = MLStripper()
        s.feed(html)
        return s.get_data()

    def unescapeHtmlChars(self, s):
        """
        Replaces : ;lt; ;gt; ;amp; ;apos; ;quot; in text with ""
        :param s: string content
        :return: clean string without above mentioned strings
        """
        s = s.replace(";lt;", "").replace(";gt;", "").replace(";amp;", "").replace(";apos;", "").replace(";quot;", "")
        return s

    def process_html_content(self, line):
        """
        Parses HTML and removes HTML charachters.
        :param line:
        :return:
        """
        parser = HTMLParser()
        line = parser.unescape(self.unescapeHtmlChars(line.strip()))
        processed_output = self.strip_tags(line) + "\n"
        return processed_output.encode("utf-8")


class Word2VecTextNormalizer(object):
    def __init__(self):
        self.pc = ProcessHTMLContent()
        self.rgx = re.compile('[^\wäöüß€\n.$]', re.UNICODE)
        self.multiple_dots_pattern = re.compile(r'(\.+)')  # ........
        self.multiple_spaces_pattern = re.compile(r'(\s+)')  # "      "


    def remove_non_alphanumchars(self, word):
        """
        Removes non alphanumeric characters from the word based on regex match.

        :param word: word in a line of text
        :return: word as string with alphanumeric characters and german characters
        """
        out_line = re.sub(self.rgx, ' ', word)
        return out_line

    def grouper(self, n, iterable, padvalue=None):
        """
        Groups a list into a list of n elements per item
        Example:
        grouper(3, 'abcdefg', 'x') -->
        ('a','b','c'), ('d','e','f'), ('g','x','x')"""
        return izip_longest(*[iter(iterable)] * n, fillvalue=padvalue)

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

    def remove_stop_words(self, line):
        """
        TODO: Stop word removal for both English and German language using standard NLTK.
        :return: text as string without any stop words
        """

        pass

    def convert_line_unicode(self, line):
        """ UTF-8 encodes a string (if not already) and returns the encoded string"""
        try:
            if not isinstance(line, unicode):
                line = line.encode("utf-8")
        except:
            pass
        return line

    def preprocess_line(self, line):
        try:

            if line.strip() == "":
                return line

            # Process HTML Content
            line = self.convert_line_unicode(line)

            line = self.pc.process_html_content(line)

            # Remove New line and Tab characters
            line = line.replace("\n", "").replace("\r", "").replace("\t", "")

            # Replace the one and more occurrences of '.' with a single 'full_stop'.
            line_dots_rep = re.sub(self.multiple_dots_pattern, '.', line)

            # Replace full stop with: a full stop with spaces around it.
            line = line_dots_rep.replace(".", " . ")
            # --> modified to remove dots all-together
            # line = line_dots_rep.replace(".", "")

            # Remove non-Alphanumeric characters in a word and rejoin the string
            line = ' '.join([self.remove_non_alphanumchars(x.strip().decode("utf-8").lower())
                             for x in line.split(" ")])
            # remove multiple spaces with a single space
            line = re.sub(self.multiple_spaces_pattern, ' ', line)

            return line
        except Exception as e:
            print("Word2Vec Normalizer Error: {}, Line: {}".format(e, line))

