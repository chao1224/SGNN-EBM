import os
import pickle
from collections import defaultdict
from step_03 import assay_to_uniprot
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chembl_raw')
args = parser.parse_args()
root = '../{}/raw/'.format(args.dataset)


if __name__ == '__main__':
    assay2uniprot = assay_to_uniprot()
    print(list(assay2uniprot.keys())[:10])

    assay_list = list(assay2uniprot.keys())
    assay2id = {a:i for i,a in enumerate(assay_list)}

    f1 = open(os.path.join('filtered_assay_score.tsv'), 'r')
    f2 = open(os.path.join('filtered_task_score.tsv'), 'w')
    assay_pair_score = {}
    for line in f1:
        line = line.strip().split('\t')
        a1, a2, score = line[0], line[1], float(line[2])
        print('{}\t{}\t{}'.format(assay2id[a1], assay2id[a2], score), file=f2)
