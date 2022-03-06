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
    '''
    TODO: need to double check
    CHEMBL1002712
    CHEMBL1022010
    CHEMBL1024480
    CHEMBL1026933
    CHEMBL1026934
    CHEMBL1031359
    CHEMBL1033994
    '''

    assay_list = list(assay2uniprot.keys())
    assay2id = {a:i for i,a in enumerate(assay_list)}

    f1 = open(os.path.join('filtered_assay_score.tsv'), 'r')
    f2 = open(os.path.join('filtered_task_score.tsv'), 'w')
    assay_pair_score = {}
    for line in f1:
        line = line.strip().split('\t')
        a1, a2, score = line[0], line[1], float(line[2])
        print('{}\t{}\t{}'.format(assay2id[a1], assay2id[a2], score), file=f2)

    # root = '../../chem/dataset/chembl_raw/raw/'
    # f = open(os.path.join(root, 'labelsHard.pckl'), 'rb')
    # targetMat = pickle.load(f)
    # print('targetMat\t', targetMat)
    #
    # print(type(targetMat))
    # print(targetMat.indices, '\t', targetMat.indptr)
    # print(len(targetMat.indices), '\t', len(targetMat.indptr))
