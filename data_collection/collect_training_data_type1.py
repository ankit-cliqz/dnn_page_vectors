#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Type 1 : Collecting Gold Data from the type 1 queries collected from cliqz-search
Sample Data:

{"q":"center city clarion collection frankfurt hotel","doc_incorr":{"http:\/\/nifox.de\/ibanreverse":"iban pr\u00fcfer welche bank iban iban de15400400 welche bank iban welche bank welche bank ist das ibanpr\u00fcfer bankleitzahl iban pr\u00fcfer alle banken deutschlands bankleitzahlensuche findet bankleitzahlen oder banken nach bankname bankleitzahl ort oder postleitzahl unterst\u00fctzt unvollst\u00e4ndige angaben und zahlen","http:\/\/www.idealo.de\/preisvergleich\/ProductCategory\/2160F1113656.html":"geschirrsp\u00fcler vollintegrierbar vollintegrierbarer geschirrsp\u00fcler vollintegrierter geschirrsp\u00fcler geschirrsp\u00fcler vollintegriert geschirrsp\u00fcler integrierbar sp\u00fclmaschine vollintegrierbar  ","https:\/\/www.siteground.com\/tutorials\/email\/pop3-imap-smtp-ports.htm":"imap port pop3 port smtp port smtp ports port 995 port pop3 email protocols pop3 smtp and imap ports and configuration find out more about the e mail protocols pop3 smtp and imap which are their default ports and how to configure them"},"doc_corr":{"https:\/\/www.tripadvisor.de\/Hotel_Review-g187337-d617952-Reviews-Clarion_Collection_Hotel_Frankfurt_City-Frankfurt_Hesse.html":"clarion collection hotel frankfurt city clarion collection hotel frankfurt city center clarion collection hotel frankfurt clarion collection hotel clarion collection clarion collection frankfurt city clarion collection hotel frankfurt city frankfurt am main 96 hotel bewertungen clarion collection hotel frankfurt city frankfurt am main 96 bewertungen 68 authentische reisefotos und g\u00fcnstige angebote f\u00fcr clarion collection hotel frankfurt city bei tripadvisor auf platz 149 von 266 hotels in frankfurt am main mit 3 5 von reisenden bewertet"}}

"""
import ujson as json

print "Starting ... "
input_file = "model_training_data_urls.txt"

output_file = open("cliqz_type1_gold_data.txt" , "w")
urls_file = open("cliqz_type1_urls_all.txt" , "w")

urls_all = set()
with open(input_file, "r") as fo:
    for line in fo:
        doc = json.loads(line)
        query = doc['q']
        incorr_doc_dict = doc['doc_incorr']
        corr_doc_dict = doc['doc_corr']
        incorr_doc_list = []
        corr_doc = ""

        for k1,v1 in incorr_doc_dict.iteritems():
            incorr_doc_list.append(k1)
            urls_all.add(k1)

        for k2,v2 in corr_doc_dict.iteritems():
            corr_doc = k2
            urls_all.add(corr_doc)
            break

        output_json = {
            'q': query,
            'corr_url': corr_doc,
            'incorr_url': incorr_doc_list
        }

        output_file.write(json.dumps(output_json) + "\n")



output_file.close()

for url in urls_all:
    urls_file.write(json.dumps(url)+"\n")

urls_file.close()
print "Finished"









