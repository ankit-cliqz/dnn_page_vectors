#!/usr/bin/python
# -*- coding: UTF-8 -*-


"""
Parsing Google Type-3 dataset for gold data collection in format: <q> <correct_url> <incorrect_url_list>
Sample Data from Dataset:
hauptsitz acal bfi      http://www.acalbfi.com/de http://www.acalbfi.com/de/services/contact-us http://www.acalbfi.com/de/aboutAcal https://www.xing.com/companies/acalbfigermanygmbh https://www.wer-zu-wem.de/firma/bfi-optilas.html
five ten spitfire 2015  http://www.bike-discount.de/de/five-ten http://www.hibike.de/five-ten-schuhe-mtb-mg70569-960 https://www.amazon.com/Five-Ten-Mens-Spitfire-Bike/dp/B008XES5TI

"""

import ujson as json
import random

input_file = "data_type3_in_index.txt"

url_set = set()
with open(input_file, "r") as fo:
    for line in fo:
        component = line.strip().split("\t")
        query = component[0]
        url_list = component[1].split(" ")
        for url in url_list:
            url_set.add(url)


print "All url's collected! Now writing to file ... "
urls_file = open("all_urls_google_type3.txt", "w")
for url in url_set:
    urls_file.write("{}\n".format(json.dumps(url)))
urls_file.close()

print "All url's written to file!"
print "Collecting gold data ...: <q> <correct_url> <incorrect_url_list>."
output_file = open("google_type3_gold_data.txt", "w")

def get_incorrect_url_list(url_set, corr_url_list):
    incorr_list = []
    while len(incorr_list) < 3:
        random_url = random.sample(url_set, 1)
        if not random_url in set(corr_url_list) and not random_url in set(incorr_list):
            incorr_list.append(random_url)
    return incorr_list


with open(input_file, "r") as fo:
    for line in fo:
        component = line.strip().split("\t")
        query = component[0]
        corr_url_list = component[1].split(" ")
        for u in corr_url_list:
            for i in xrange(3):
                incorr_url_tmp = get_incorrect_url_list(url_set, corr_url_list)
                output_json = {
                    'q': query,
                    'corr_url': u,
                    'incorr_url': incorr_url_tmp
                }
                output_file.write(json.dumps(output_json) + "\n")

output_file.close()
print "Finished ... "