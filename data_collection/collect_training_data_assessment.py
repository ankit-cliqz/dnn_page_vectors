#!/usr/bin/python
# -*- coding: UTF-8 -*-
###
# Parsing UCrawl Assessment Data for collecting Gold Data with a given query correct document and incorrect document pairs.
###

import ujson as json
import random


input_file = "UCrawl_tillWk34.json"
ass_list = ['Vital', 'Useful', 'Relevant', 'Slightly relevant', 'Off-topic / Useless', 'Foreign language', 'Not found']

url_set_all = set()
print "Collecting all URLS in a set ... "

with open(input_file, "r") as fo:
    for line in fo:
        data = json.loads(line)
        res = data['results']
        for item in res:
            url = item['url']
            url_set_all.add(url)

print "All url's collected! Now writing to file ... "
urls_file = open("all_urls_assessment.txt", "w")
for url in url_set_all:
    urls_file.write("{}\n".format(json.dumps(url)))
urls_file.close()

def get_random_incorr_url_list(corr_url, incorr_url, url_set_all):
    incorr_url_tmp = []
    # Generate a new list of negative examples for each correct url per query
    while (not len(incorr_url_tmp) == 3):
        for url_tmp in incorr_url:
            incorr_url_tmp.append(url_tmp)
        if len(incorr_url_tmp) < 3:
            random_url = random.sample(url_set_all, 1)
            if random_url not in corr_url and random_url not in incorr_url:
                incorr_url_tmp.append(random_url)
    return incorr_url_tmp


print "All url's written to file!"
print "Collecting gold data ...: <q> <correct_url> <incorrect_url_list>."
output_file = open("assessment_gold_data_enriched.txt", "w")
with open(input_file, "r") as fo:
    for line in fo:
        data = json.loads(line)
        query = data['query']
        res = data['results']

        url_list_with_ass_rank = []
        for item in res:
            url = item['url']
            assessment = item['assessment']
            url_list_with_ass_rank.append((url, ass_list.index(assessment)))
            # print query, url, assessment
        ul_sorted = sorted(url_list_with_ass_rank, key=lambda x: x[1])
        corr_url = []
        incorr_url = []
        for url, ass_id in ul_sorted:
            if ass_id < 4:
                corr_url.append(url)
            elif ass_id > 3 and not ass_id == 5:
                incorr_url.append(url)

        if not len(corr_url) == 0:
            for u in corr_url:
                if len(incorr_url) == 3:
                    incorr_url_tmp = get_random_incorr_url_list(corr_url, incorr_url, url_set_all)
                    output_json = {
                        'q': query,
                        'corr_url':u,
                        'incorr_url':incorr_url_tmp
                    }
                    output_file.write(json.dumps(output_json)+"\n")
                else:
                    for i in xrange(0,3):
                        incorr_url_tmp = get_random_incorr_url_list(corr_url, incorr_url, url_set_all)
                        output_json = {
                            'q': query,
                            'corr_url': u,
                            'incorr_url': incorr_url_tmp
                        }
                        output_file.write(json.dumps(output_json) + "\n")


print "Finished!"