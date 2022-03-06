from collections import OrderedDict, defaultdict
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chembl_raw')
args = parser.parse_args()
root = '../{}/raw/'.format(args.dataset)


def assay_to_uniprot():
    assay2uniprot = OrderedDict()
    f = open(os.path.join('assay2target.tsv'), 'r')
    f.readline()
    for line in f:
        line = line.strip().split('\t')
        assay = line[0]
        uniprot_list = line[-1].split(',')
        uniprot_list = list(filter(lambda x: len(x)==6, uniprot_list))
        assay2uniprot[assay] = uniprot_list
    return assay2uniprot


def uniprot_to_string():
    uniprot2string = {}
    f = open('uniprot2string.tsv', 'r')
    for line in f:
        line = line.strip().split('\t')
        uniprot, string_id = line
        uniprot2string[uniprot] = string_id
    return uniprot2string


def get_string_pair_score():
    f = open('string_ppi_score.tsv', 'r')
    record = defaultdict(int)
    for line in f:
        line = line.strip().split('\t')
        p1, p2, score = line[0], line[1], float(line[2])
        if not p1.startswith('9606'):
            p1, p2 = '9606.{}'.format(p1), '9606.{}'.format(p2)
        if (p1, p2) in record:
            print(line)
        record[(p1, p2)] = score
        record[(p2, p1)] = score
    print('{} records in string pair'.format(len(record)))
    print()
    return record


def map_from_uniprot_list_to_string(uniprot_list, uniprot2string):
    ret = []
    for u in uniprot_list:
        if u in uniprot2string:
            ret.append(uniprot2string[u])
    return ret


def get_assay_pair_score(assay_i, assay_j, assay2uniprot, uniprot2string, string_pair_record):
    uniprot_list_i = assay2uniprot[assay_i]
    uniprot_list_j = assay2uniprot[assay_j]
    string_list_i = map_from_uniprot_list_to_string(uniprot_list_i, uniprot2string)
    string_list_j = map_from_uniprot_list_to_string(uniprot_list_j, uniprot2string)
    score = 0
    for si in string_list_i:
        for sj in string_list_j:
            score = max(score, string_pair_record[(si, sj)])
    return score


if __name__ == '__main__':
    assay2uniprot = assay_to_uniprot()
    uniprot2string = uniprot_to_string()
    string_pair_record = get_string_pair_score()

    for k, v in assay2uniprot.items():
        print(k, v)
    print()

    f = open(os.path.join('filtered_assay_score.tsv'), 'w')
    assay_list = list(assay2uniprot.keys())
    N = len(assay_list)
    for i in range(N):
        assay_i = assay_list[i]
        for j in range(i+1, N):
            assay_j = assay_list[j]

            score = get_assay_pair_score(assay_i, assay_j, assay2uniprot, uniprot2string, string_pair_record)
            if score > 0:
                print(assay_i, '\t', assay_j, '\t', score)
                print('{}\t{}\t{}'.format(assay_i, assay_j, score), file=f)
