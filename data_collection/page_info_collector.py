#!/usr/bin/python
# -*- coding: UTF-8 -*-


#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
 --> Collect training data for the Query-Document Similarity LSTM

"""

import ujson as json
# import pykeyvi
from cache.db.sm import SM
from cache.db.word2vec_normalizer import Word2VecTextNormalizer
from cache.db.data_utils import DataUtils

uc = 0                                  # url_ids not matching
nf=0
sources = ["ucrawl", "qs", "qc"]        # Top Query Sources
top_n = 6                               # Top - n - queries

sm = SM()
du = DataUtils()
wn = Word2VecTextNormalizer()

url_prefix_list = ["http://", "https://", "http://www.", "https://www."]
def get_top_queries_source(data_json, top_n_queries, source):
    if (not data_json["tq"].get(source) is None):
        queries_source_list = data_json["tq"][source]
        if len(queries_source_list) > 1:
            queries_source_list.sort(key=lambda tup: tup[1])
            queries_source_list.reverse()
        item_count = 0
        for data_item in queries_source_list:
            if item_count == top_n:
                break
            item_count += 1
            if (not data_item[0].strip().replace(" ", "").isdigit()):
                top_n_queries.append(data_item[0].strip())


url_set = set()
with open("/ebs/ankit/all_urls_unique.txt", "r") as fr:
    for line in fr:
        line = json.loads(line)
        if not line.strip() == "" and line.startswith('http'):
            url_set.add(str(line.strip()))

out_file = open("/raid/page_info_collected_urls.txt", "w")
# keyvi_index_compiler = pykeyvi.JsonDictionaryCompiler()
ucount = 0

for url in url_set:
    try:
        ucount+=1
        if ucount % 10000 == 0:
            print "Currently Processed: {}".format(ucount)
        correct_url_found = False
        # Handle the case for missing 'web-url' prefixes
        url_tmp = url
        #print "URL: {}".format(url)
        #print "URL Type: {}".format(type(url))
        data = sm.info_page(url_tmp)
        #print data
        #print "Data Type: {}".format(type(data))
        if data['info'] and not data['info'] is None:
            if not data['info'] == 'not in index':
                correct_url_found = True


        # Handle the case for missing training '/' in a web url for the cases only
        # where the url was not found in the index
        if correct_url_found == False:
            #for prefix in url_prefix_list:
            url_tmp = url + '/'
            # print "URL: {}".format(url)
            # print "URL Type: {}".format(type(url))
            data = sm.info_page(url_tmp)
            # print "Data Type: {}".format(type(data))
            if data['info'] and not data['info'] is None:
                if not data['info'] == 'not in index':
                    correct_url_found = True

        if correct_url_found == False:
            nf+=1
            print "URL NOT FOUND IN INDEX: {}".format(url)
            continue

        #print "yes"
        data = sm.info_page(url_tmp)
        #print data
        if data['info'] and not data['info'] is None:
            top_n_queries = []

            # Collect top queries linked to page
            if (not data['info'].get("tq") is None):
                for s in sources:
                    get_top_queries_source(data['info'], top_n_queries, s)

            url_id_data = ""
            if data['url_id'] is not None:
                url_id_data = data['url_id']


            title = ""
            desc = ""

            # print url_tmp
            if data['snippets'] and not data['snippets'] == 'no snippet info':
                if data['snippets']['snippet']:
                        if data['snippets']['snippet'].get('title') is not None:
                            title = data['snippets']['snippet']['title']
            else:
                title = ""

            if data['snippets'] and not data['snippets'] == 'no snippet info' :
                if data['snippets']['snippet']:
                        if data['snippets']['snippet'].get('desc') is not None:
                            desc = data['snippets']['snippet']['desc']
            else:
                desc = ""

            url_words = " ".join([wn.preprocess_line(du.extract_domain(url_tmp))] + du.words_from_url(url_tmp, ordered_list=True))
            if not len(top_n_queries) == 0:
                data_out = {
                    'url_words': url_words,
                    'title': wn.preprocess_line(title),
                    'desc': wn.preprocess_line(desc),
                    'top_n_q': [wn.preprocess_line(q) for q in top_n_queries[:top_n]]
                }
                # print data_out
            else:
                data_out = {
                    'url_words': url_words,
                    'title': wn.preprocess_line(title),
                    'desc': wn.preprocess_line(desc),
                    'top_n_q': []
                }
                # print data_out

            out_file.write("{}\t{}\n".format(json.dumps(url_tmp), json.dumps(data_out)))
            #keyvi_index_compiler.Add(str(url_tmp), json.dumps(data_out))
    except Exception as e:
        print "Exception encountered for url: {}, exception: {}".format(url, e)

out_file.close()
print "Number of Total URLS: {}".format(len(url_set))
print "Number of url-ids mis-match URLS: {}".format(uc)
print "Number of URLS not found in index: {}".format(nf)
# print "\nCompiling Keyvi Index < url:id - url for training data >... "
# keyvi_index_compiler.Compile()
# print "Finished Compiling Keyvi. Now writing keyvi index to file... "
# keyvi_index_compiler.WriteToFile('/raid/train_data_for_urls.kv')
print "Finished writing keyvi file!"
