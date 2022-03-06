import pickle
import os
from multiprocessing import Pool
from collections import defaultdict, OrderedDict
import urllib
from urllib.error import HTTPError
import urllib.request
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chembl_raw')
parser.add_argument('--n-proc', type=int, default=12, help='number of processes to run when downloading assay & target information')
args = parser.parse_args()


def get_record(website):
    try:
        data = urllib.request.urlopen(website).read().decode("utf-8")
    except HTTPError:
        return {}
    root = ET.fromstring(data)
    record = defaultdict(list)
    for child in root:
        record[child.tag] = child.text

    for item in root.findall('target_components/target_component/target_component_xrefs/target'):
        xref_id = item.find('xref_id').text
        xref_src_db = item.find('xref_src_db').text
        record[xref_src_db].append(xref_id)

    return record


def generate_table_row_str(assay_id):
    assay_record = get_record('https://www.ebi.ac.uk/chembl/api/data/assay/{}'.format(assay_id))
    if not assay_record:
        return '\t\t\t\t'
    target_id = assay_record['target_chembl_id']
    target_record = get_record('https://www.ebi.ac.uk/chembl/api/data/target/{}'.format(target_id))
    target_name = target_record['pref_name']
    organism = target_record['organism']
    uniprot_list = target_record['UniProt']
    return '{}\t{}\t{}\t{}\t{}'.format(
        assay_id,
        target_id,
        target_name,
        organism,
        ','.join(uniprot_list)
    )


if __name__ == '__main__':
    root = '../{}/raw/'.format(args.dataset)
    with open(os.path.join(root, 'labelsHard.pckl'), 'rb') as f:
        targetMat = pickle.load(f)
        sampleAnnInd = pickle.load(f)
        targetAnnInd = pickle.load(f)

    assay_ids = list(targetAnnInd.index)
    with Pool(args.n_proc) as p:
        table_rows_str = p.map(generate_table_row_str, tqdm(assay_ids))
    with open('assay2target.tsv', 'w') as g:
        g.write('assay_id\ntarget_id\ttarget_name\norganism\tuniprot_list\n')
        g.write('\n'.join(table_rows_str))
