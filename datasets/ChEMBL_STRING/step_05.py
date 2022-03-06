import pickle


def print_target_name(target_name):
    for x in target_name:
        print(x, end='\t')
    print()
    print()
    print()
    return


### Thanks Zuobai for the discussion! :)
if __name__ == '__main__':
    dataPathSave = '../chembl_raw/raw/'

    f=open(dataPathSave + 'folds0.pckl', 'rb')
    folds=pickle.load(f)
    f.close()
    print('folds\t', folds)
    print()

    f = open(dataPathSave + 'labelsHard.pckl', 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()
    print('targetAnnInd\t', targetAnnInd)
    print('targetMat\t', targetMat)
    print('targetAnnInd\t', targetAnnInd)
    # print('targetAnnInd')
    # target_name = targetAnnInd.index.values
    # print_target_name(target_name)
    print()

    f = open(dataPathSave + 'labelsWeakHard.pckl', 'rb')
    targetMatWeak = pickle.load(f)
    sampleAnnIndWeak = pickle.load(f)
    targetAnnIndWeak = pickle.load(f)
    f.close()
    print('targetAnnInd\t', targetAnnInd)
    # print('targetAnnIndWeak')
    # target_name = targetAnnIndWeak.index.values
    # print_target_name(target_name)
