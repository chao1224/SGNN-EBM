import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols) + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    return train_idx, valid_idx, test_idx


def assign_pos_or_neg_to_data_idx(candidate_idx, data_idx, selected_idx):
    for idx in candidate_idx:
        if idx in data_idx:
            return True

    for idx in candidate_idx:
        if idx not in selected_idx:
            selected_idx.add(idx)
            data_idx.add(idx)
            return True

    return False


def random_filtered_trial(task_count, all_idx, task_idx, y_task, valid_len, test_len):
    train_idx, valid_idx, test_idx = set(), set(), set()
    selected_idx = set()

    for t in task_idx:
        pos_idx = task_count[(t, 1)]
        neg_idx = task_count[(t, -1)]
        random.shuffle(pos_idx)
        random.shuffle(neg_idx)

        possible = assign_pos_or_neg_to_data_idx(pos_idx, train_idx, selected_idx)
        if not possible:
            return False, [], [], []
        possible = assign_pos_or_neg_to_data_idx(neg_idx, train_idx, selected_idx)
        if not possible:
            return False, [], [], []

        possible = assign_pos_or_neg_to_data_idx(pos_idx, valid_idx, selected_idx)
        if not possible:
            return False, [], [], []
        possible = assign_pos_or_neg_to_data_idx(neg_idx, valid_idx, selected_idx)
        if not possible:
            return False, [], [], []

        possible = assign_pos_or_neg_to_data_idx(pos_idx, test_idx, selected_idx)
        if not possible:
            return False, [], [], []
        possible = assign_pos_or_neg_to_data_idx(neg_idx, test_idx, selected_idx)
        if not possible:
            return False, [], [], []

    train_idx, valid_idx, test_idx = list(train_idx), list(valid_idx), list(test_idx)
    random.shuffle(all_idx)
    print('Done ============')

    valid_len -= len(valid_idx)
    test_len -= len(test_idx)
    for i in all_idx:
        if i in selected_idx:
            continue
        selected_idx.add(i)
        if valid_len > 0:
            valid_len -= 1
            valid_idx.append(i)
        elif test_len > 0:
            test_len -= 1
            test_idx.append(i)
        else:
            train_idx.append(i)

    return True, train_idx, valid_idx, test_idx


def random_filtered_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0, num_trial=10):
    '''
    Make sure each task has at least one pos and on neg after data splitting.
    '''
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    y_task = np.array([data.y.numpy() for data in dataset])

    num_mols, num_tasks = y_task.shape
    random.seed(seed)

    valid_len = int(frac_valid * num_mols)
    test_len = int(frac_test * num_mols)

    task_count = {}
    task_idx = []
    for i in range(num_tasks):
        label = y_task[:, i]
        task_count[(i, 1)] = np.nonzero(label == 1)[0]
        task_count[(i, -1)] = np.nonzero(label == -1)[0]
        task_idx.append([i, len(task_count[(i, 1)]), len(task_count[(i, -1)])])

    ########## Sort w.r.t. task (positive data) count ##########
    task_idx.sort(key=lambda x: x[1])
    task_idx = [idx for [idx, _, _] in task_idx]
    all_idx = np.arange(num_mols)

    valid, train_idx, valid_idx, test_idx = False, [], [], []
    for trial_idx in range(num_trial):
        print('trial {}'.format(trial_idx))
        valid, train_idx, valid_idx, test_idx = random_filtered_trial(task_count, all_idx, task_idx, y_task, valid_len, test_len)
        if valid:
            break

    if not valid:
        print('Unable to split {}-{}-{} with seed {}.'.format(frac_train, frac_valid, frac_test, seed))
        return random_split(dataset, frac_train, frac_valid, frac_test, seed)

    train_idx = [int(i) for i in train_idx]
    valid_idx = [int(i) for i in valid_idx]
    test_idx = [int(i) for i in test_idx]

    print('Being able to split {}-{}-{} with seed {}.'.format(frac_train, frac_valid, frac_test, seed))
    return train_idx, valid_idx, test_idx


def random_scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
                          col_name='scaffold_smiles', root='../dataset/chembl_full'):
    """
    scaffold_smiles: split by Bemis-Murcko scaffolds
    clusterID: split by clusters defined by single-linkage hierarchical clustering
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    num_mols = len(dataset)
    mol_prop = pd.read_csv(os.path.join(root, 'mol_properties.csv'), index_col=0)
    scaffold_to_mols = defaultdict(list)
    for i, scaffold in mol_prop[col_name].items():
        scaffold_to_mols[scaffold].append(i)
    unique_scaffolds = list(scaffold_to_mols.keys())
    random.seed(seed)
    random.shuffle(unique_scaffolds)

    train_max = int(frac_train * num_mols)
    valid_max = int((frac_valid + frac_train) * num_mols)

    train_idx, valid_idx, test_idx = [], [], []
    i = 0
    for scaffold in unique_scaffolds:
        scaffold_ls = scaffold_to_mols[scaffold]
        if i < train_max:
            train_idx.extend(scaffold_ls)
        elif i < valid_max:
            valid_idx.extend(scaffold_ls)
        else:
            test_idx.extend(scaffold_ls)
        i += len(scaffold_ls)
    print(f'Split by scaffold: train {len(train_idx)}, valid {len(valid_idx)}, test {len(test_idx)}')
    return train_idx, valid_idx, test_idx