import pandas as pd
import requests
import urllib
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from tqdm import tqdm
from time import sleep
from requests.models import HTTPError


'''
http://www.uniprot.org/uniprot/O75713
http://www.uniprot.org/uniprot/D3DTF2


https://string-db.org/api/tsv/get_string_ids?identifiers=D3DTF2

https://string-db.org/api/json/network?identifiers=[your_identifiers]&[optional_parameters]

check this: https://string-db.org/cgi/access
'''

parser = argparse.ArgumentParser()
parser.add_argument('--n-proc', type=int, default=12, help='number of processes to run when downloading assay & target information')
args = parser.parse_args()


def mapping_to_string_API(valid_string_set):
    string_api_url = "https://version-11-0.string-db.org/api"
    output_format = "tsv-no-header"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])
    print('request_url\t', request_url)

    valid_string_set = list(valid_string_set)
    params = {
        "identifiers": "%0d".join(valid_string_set),  # your protein
        "species": 9606,  # species NCBI identifier
    }
    print('len of genes\t', len(valid_string_set))

    response = requests.post(request_url, data=params)
    print(response)

    with open('string_ppi_score.tsv', 'w') as string_ppi_file:
        pair_count, pos_pair_count = 0, 0
        for line in response.text.strip().split("\n"):
            l = line.strip().split("\t")
            p1, p2 = '{}'.format(l[0]), '{}'.format(l[1])
            experimental_score = float(l[10])
            # print("\t".join([p1, p2, "experimentally confirmed (prob. %.3f)" % experimental_score]))
            print('{}\t{}\t{}'.format(p1, p2, experimental_score), file=string_ppi_file)
            pair_count += 1
            if experimental_score > 0.2:
                pos_pair_count += 1
    print(pair_count, '\t', pos_pair_count)
    print()


def query_stringid(uniprot):
    website = 'https://version-11-0.string-db.org/api/xml/get_string_ids?identifiers={}'.format(uniprot)
    try:
        with urllib.request.urlopen(website) as conn:
            data = conn.read().decode("utf-8")
    except HTTPError:
        data = ''
    if data:
        root = ET.fromstring(data)
        string_id_result = root.find('record/stringId')
        if string_id_result is not None:
            return string_id_result.text

    print('error on {}: {}'.format(uniprot, data))
    return ''


def store_mapping_from_uniprot_to_string_id(uniprot_set):
    print('Storing mapping from uniprot to string to uniprot2string.tsv...')
    with Pool(args.n_proc) as p:
        string_id_set = p.map(query_stringid, tqdm(uniprot_set))
    num_errors = 0
    with open('uniprot_without_strid.txt', 'w') as r, open('uniprot2string.tsv', 'w') as g:
        for uniprot, string_id in zip(uniprot_set, string_id_set):
            if string_id:
                g.write('{}\t{}\n'.format(uniprot, string_id))
            else:
                r.write(uniprot + '\n')
                num_errors += 1
    print('Done storing. Number of errors: {}. Mapped uniprots: {}'.format(num_errors, len(uniprot_set) - num_errors))


if __name__ == '__main__':
    '''
    Assay ID\tTarget ID\tTarget Name\tOrganism\t{UniProt list}
    '''
    assay2target_fname = 'assay2target.tsv'
    uniprot_set = set()
    with open(assay2target_fname, 'r') as assay2target_file:
        assay2target_file.readline()
        for line in assay2target_file:
            line = line.strip().split('\t')
            uniprot_list = line[-1].strip().split(',')
            # print(uniprot_list)
            for uniprot in uniprot_list:
                if len(uniprot) != 6:
                    continue
                uniprot_set.add(uniprot)

    store_mapping_from_uniprot_to_string_id(uniprot_set)
    
    with open('uniprot2string.tsv', 'r') as uniprot2string_file:
        valid_string_set = set()
        for line in uniprot2string_file:
            line = line.strip().split('\t')
            uniprot = line[0]
            string_id = line[1]
            valid_string_set.add(string_id)

    mapping_to_string_API(valid_string_set)
