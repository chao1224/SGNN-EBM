import sys
sys.path.insert(0, '../src')
from main_SGNN_EBM import *


def get_task_representation(task_embedding_model, kg_model, task_X):
    task_repr = task_embedding_model(task_X)
    if args.use_GCN_for_KG:
        task_repr = kg_model(task_repr, task_edge)
    return task_repr


def eval(args, model, device, loader, evaluation_mode):
    model.eval()
    task_embedding_model.eval()
    first_order_prediction_model.eval()
    if second_order_prediction_model is not None:
        second_order_prediction_model.eval()
    if GNN_energy_model is not None:
        GNN_energy_model.eval()
    if args.use_GCN_for_KG:
        kg_model.eval()

    y_true_list = []
    y_pred_list = []
    id_list = []

    with torch.no_grad():
        task_repr = get_task_representation(task_embedding_model, kg_model, task_X)

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            graph_repr = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            id_list.append(batch.id.cpu())

            B = len(graph_repr)
            y_true = batch.y.view(B, args.num_tasks).float()

            if args.inference_function in [
                'GNN_1st_order_inference',
                'GNN_EBM_mean_field_variational_inference',
                'GNN_EBM_GS_inference',
                'GNN_EBM_1st_order_inference_Binary_Task',
            ]:
                y_pred = inference_function(
                    first_order_prediction_model=first_order_prediction_model,
                    second_order_prediction_model=second_order_prediction_model,
                    GNN_energy_model=GNN_energy_model,
                    prediction_function=prediction_function,
                    graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                    prior_prediction=prior_prediction, id=batch.id,
                    prior_prediction_logits=prior_prediction_logits,
                    args=args)
            else:
                y_pred = inference_function(
                    first_order_prediction_model=first_order_prediction_model,
                    second_order_prediction_model=second_order_prediction_model,
                    energy_function=energy_function,
                    graph_repr=graph_repr, task_repr=task_repr, y_true=y_true, task_edge=task_edge,
                    first_order_label_weights=first_order_label_weights, second_order_label_weights=second_order_label_weights,
                    args=args)

            y_true_list.append(y_true.cpu())
            y_pred_list.append(y_pred.cpu())

        id_list = torch.cat(id_list, dim=0).numpy()
        y_true_list = torch.cat(y_true_list, dim=0).numpy()
        y_pred_list = torch.cat(y_pred_list, dim=0).numpy()

        roc_list = []
        invalid_count = 0
        for i in range(y_true_list.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true_list[:, i] == 1) > 0 and np.sum(y_true_list[:, i] == -1) > 0:
                is_valid = y_true_list[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true_list[is_valid, i] + 1) / 2, y_pred_list[is_valid, i]))
            else:
                invalid_count += 1

        print('Invalid task count:\t', invalid_count)

        if len(roc_list) < y_true_list.shape[1]:
            print('Some target is missing!')
            print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true_list.shape[1]))

        roc_list = np.array(roc_list)
        roc_value = np.mean(roc_list)
    return roc_value


def extract_prior_distribution_from_pretrained_model(args):
    prior_data = np.load('ebm/{}/{}/{}.npz'.format(args.dataset, args.seed, "model_prior"))
    prior_pred, prior_pred_logits = prior_data["prior_prediction"], prior_data["prior_prediction_logits"]
    prior_pred = torch.LongTensor(prior_pred).to(args.device)
    prior_pred_logits = torch.FloatTensor(prior_pred_logits).to(args.device)
    return prior_pred, prior_pred_logits


def load_model(output_model_file):
    ckpt = torch.load('{}/{}/model_best.pth'.format(output_model_file, args.seed))
    print('Loading from {} ...\n'.format(output_model_file))
    print(ckpt.keys())

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    if 'task_embedding_model' in ckpt:
        task_embedding_model.load_state_dict(ckpt['task_embedding_model'])
    if 'first_order_prediction_model' in ckpt:
        first_order_prediction_model.load_state_dict(ckpt['first_order_prediction_model'])
    if 'second_order_prediction_model' in ckpt:
        second_order_prediction_model.load_state_dict(ckpt['second_order_prediction_model'])
    if 'GNN_energy_model' in ckpt:
        GNN_energy_model.load_state_dict(ckpt['GNN_energy_model'])
    if 'kg_model' in ckpt:
        kg_model.load_state_dict(ckpt['kg_model'])
    # This is only used for pre-training
    # if 'readout_model' in ckpt:
    #     readout_model.load_state_dict(ckpt['readout_model'])
    # if 'task_relation_matching_model' in ckpt:
    #     task_relation_matching_model.load_state_dict(ckpt['task_relation_matching_model'])
    # if 'NCE_C_param' in ckpt:
    #     NCE_C_param.load_state_dict(ckpt['NCE_C_param'])

    return


if __name__ == '__main__':
    print(args.num_tasks)
    print('arguments\t', args)

    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########## Set up molecule dataset ##########
    root = '../datasets/' + args.dataset
    dataset = MoleculeDataset(root=root, dataset=args.dataset)
    if args.split_method == 'random_split':
        train_indices, valid_indices, test_indices = random_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    elif args.split_method == 'random_filtered_split':
        train_indices, valid_indices, test_indices = random_filtered_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
    elif args.split_method == 'scaffold_split':
        train_indices, valid_indices, test_indices = random_scaffold_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed,
            col_name='scaffold_smiles', root=root
        )
    elif args.split_method == 'cluster_split':
        train_indices, valid_indices, test_indices = random_scaffold_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed,
            col_name='clusterID', root=root
        )
    else:
        raise ValueError('Split method {} not included.'.format(args.split_method))
    print(f'train: {len(train_indices)}\tvalid: {len(valid_indices)}\ttest: {len(test_indices)}')
    print('first train indices\t', train_indices[:10], train_indices[-10:])
    print('first valid indices\t', valid_indices[:10], valid_indices[-10:])
    print('first test indices\t', test_indices[:10], test_indices[-10:])

    train_sampler = SubsetRandomSampler(train_indices)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    valid_dataloader = test_dataloader = None
    if len(valid_indices) > 0:
        valid_sampler = SubsetRandomSampler(valid_indices)
        valid_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers)
    if len(test_indices) > 0:
        test_sampler = SubsetRandomSampler(test_indices)
        test_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.num_workers)

    ########## Set up energy function ##########
    prediction_function = None
    if args.energy_function == 'energy_function_CD_AA':
        energy_function = energy_function_CD_AA
    elif args.energy_function == 'energy_function_CD_GS':
        energy_function = energy_function_CD_GS
    elif args.energy_function == 'energy_function_GNN_CE_1st_order':
        energy_function = energy_function_GNN_CE_1st_order
        prediction_function = get_GNN_prediction_1st_order_prediction
    elif args.energy_function == 'energy_function_GNN_CE_2nd_order':
        energy_function = energy_function_GNN_CE_2nd_order
        prediction_function = get_GNN_prediction_2nd_order_prediction
    elif args.energy_function == 'energy_function_GNN_EBM_NCE':
        energy_function = energy_function_GNN_EBM_NCE
        prediction_function = get_GNN_prediction_2nd_order_prediction
    elif args.energy_function == 'energy_function_GNN_EBM_CD_GS':
        energy_function = energy_function_GNN_EBM_CD_GS
        prediction_function = get_GNN_prediction_2nd_order_prediction
    elif args.energy_function == 'energy_function_GNN_EBM_CE_2nd_order_Binary_Task':
        energy_function = energy_function_GNN_EBM_CE_2nd_order_Binary_Task
        prediction_function = get_GNN_prediction_Binary_Task_Embedding_CE
    else:
        raise ValueError('Energy function {} not included.'.format(args.energy_function))

    ########## Set up inference function ##########
    if args.inference_function == 'amortized_mean_field_inference_first_order':
        inference_function = amortized_mean_field_inference_first_order
    elif args.inference_function == 'amortized_mean_field_inference_second_order':
        inference_function = amortized_mean_field_inference_second_order
    elif args.inference_function == 'amortized_mean_field_inference_label_propagation_first_order':
        inference_function = amortized_mean_field_inference_label_propagation_first_order
    elif args.inference_function == 'mean_field_variational_inference':
        assert args.amortized_logits_transform_to_confidence
        inference_function = mean_field_variational_inference
    elif args.inference_function == 'GS_inference':
        inference_function = GS_inference
    elif args.inference_function == 'SGLD_inference':
        inference_function = SGLD_inference
    elif args.inference_function == 'GNN_1st_order_inference':
        inference_function = GNN_1st_order_inference
    elif args.inference_function == 'GNN_EBM_mean_field_variational_inference':
        inference_function = GNN_EBM_mean_field_variational_inference
    elif args.inference_function == 'GNN_EBM_GS_inference':
        inference_function = GNN_EBM_GS_inference
    elif args.inference_function == 'GNN_EBM_1st_order_inference_Binary_Task':
        inference_function = GNN_EBM_1st_order_inference_Binary_Task
    else:
        raise ValueError('Inference function {} not included.'.format(args.inference_function))

    ########## Set up assay/task embedding ##########
    task_X = torch.arange(args.num_tasks).to(device)

    ########## Set up molecule model ##########
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file + '.pth')
    model.to(device)

    ########## For assay/task embedding ##########
    if args.energy_function == 'energy_function_GNN_EBM_CE_2nd_order_Binary_Task':
        task_embedding_model = TaskEmbeddingModel_BinaryEmbedding(args.num_tasks, embedding_dim=args.task_emb_dim).to(device)
    else:
        task_embedding_model = TaskEmbeddingModel(args.num_tasks, embedding_dim=args.task_emb_dim).to(device)

    ########## For drug-protein/molecule-task prediction ##########
    first_order_prediction_model, second_order_prediction_model = None, None
    if args.energy_function in ['energy_function_CD_AA']:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=2, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=4, batch_norm=args.batch_norm).to(device)

    elif args.energy_function in ['energy_function_CD_GS']:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=2, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=4, batch_norm=args.batch_norm).to(device)

    elif args.energy_function in [
        'energy_function_GNN_CE_1st_order', 'energy_function_GNN_CE_2nd_order',
        'energy_function_GNN_EBM_NCE', 'energy_function_GNN_EBM_CD_GS',
    ]:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim*2, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim*4, batch_norm=args.batch_norm).to(device)

    elif args.energy_function in ['energy_function_GNN_EBM_CE_2nd_order_Binary_Task']:
        first_order_prediction_model = MoleculeTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim, batch_norm=args.batch_norm).to(device)
        second_order_prediction_model = MoleculeTaskTaskPredictionModel(
            args.emb_dim, args.task_emb_dim, output_dim=args.ebm_GNN_dim, batch_norm=args.batch_norm).to(device)

    # NCE_C_param = None
    # if args.energy_function == 'energy_function_GNN_EBM_NCE':
    #     NCE_C_param = NCE_C_Parameter(len(dataset)).to(device)

    ########## For GNN-EBM Model ##########
    GNN_energy_model = None
    if args.energy_function in ['energy_function_GNN_CE_1st_order']:
        if args.gnn_energy_model == 'GNN_Energy_Model_1st_Order_01':
            GNN_energy_model = GNN_Energy_Model_1st_Order_01(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat, output_dim=1).to(device)
        elif args.gnn_energy_model == 'GNN_Energy_Model_1st_Order_02':
            GNN_energy_model = GNN_Energy_Model_1st_Order_02(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat, output_dim=1).to(device)
        else:
            raise ValueError('GNN Energy Model {} not included.'.format(args.gnn_energy_model))
        # print('GNN_energy_model\n', GNN_energy_model)

    if args.energy_function in [
        'energy_function_GNN_CE_2nd_order', 'energy_function_GNN_EBM_NCE',
        'energy_function_GNN_EBM_CD_GS', 'energy_function_GNN_EBM_CE_2nd_order_Binary_Task'
    ]:
        if args.gnn_energy_model == 'GNN_Energy_Model_2nd_Order_01':
            GNN_energy_model = GNN_Energy_Model_2nd_Order_01(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat).to(device)
        elif args.gnn_energy_model == 'GNN_Energy_Model_2nd_Order_02':
            GNN_energy_model = GNN_Energy_Model_2nd_Order_02(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat).to(device)
        elif args.gnn_energy_model == 'GNN_Energy_Model_2nd_Order_03':
            GNN_energy_model = GNN_Energy_Model_2nd_Order_03(
                ebm_GNN_dim=args.ebm_GNN_dim, ebm_GNN_layer_num=args.ebm_GNN_layer_num, concat=args.ebm_GNN_use_concat).to(device)
        else:
            raise ValueError('GNN Energy Model {} not included.'.format(args.gnn_energy_model))
        # print('GNN_energy_model\n', GNN_energy_model)

    ########## Set up task-task knowledge graph dataset ##########
    ppi_dataset = PPI_dataset(args, args.PPI_threshold, neg_sample_size=args.neg_sample_size,
                              neg_sample_exponent=args.neg_sample_exponent)
    ppi_dataloader = DataLoader(ppi_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('len of PPI dataset: {}'.format(len(ppi_dataset)))
    ########## Set up task edge list / KG ##########
    task_edge = copy.deepcopy(ppi_dataset.edge_list.transpose(0, 1)).to(device) # M * 2
    ########## Set up GNN for KG ##########
    kg_model = None
    if args.use_GCN_for_KG:
        kg_model = GCNNet(embedding_dim=args.task_emb_dim, hidden_dim=args.task_emb_dim, dropout=args.kg_dropout_ratio).to(device)
    task_relation_matching_model = PairwiseTaskPredictionModel(args.task_emb_dim).to(device)

    ########## Loading prior data ##########
    prior_prediction, prior_prediction_logits = None, None
    if args.filling_missing_data_mode in ['mtl_task', 'mtl_task_KG', 'gnn']:
        prior_prediction, prior_prediction_logits = extract_prior_distribution_from_pretrained_model(args)

    load_model(args.output_model_file)

    ########## Set up 1st and 2nd order task label weights ##########
    first_order_label_weights, second_order_label_weights = extract_amortized_task_label_weights(train_dataloader, task_edge, device, args)
    first_order_label_weights = first_order_label_weights.to(device)
    second_order_label_weights = second_order_label_weights.to(device)

    # train_roc = eval(args, model, device, train_dataloader, 'train')
    valid_roc = eval(args, model, device, valid_dataloader, 'valid')
    test_roc = eval(args, model, device, test_dataloader, 'test')
    # print('train_roc\t', train_roc)
    print('valid_roc\t', valid_roc)
    print('test_roc\t', test_roc)
    print('\n')

    ########## This is for ground-truth. ##########
    test_results = np.load('ebm/{}/{}/{}.npz'.format(args.dataset, args.seed, "model_test"))
    y_true_list, y_pred_list = test_results['y_true_list'], test_results['y_pred_list']

    roc_list = []
    invalid_count = 0
    for i in range(y_true_list.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true_list[:, i] == 1) > 0 and np.sum(y_true_list[:, i] == -1) > 0:
            is_valid = y_true_list[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true_list[is_valid, i] + 1) / 2, y_pred_list[is_valid, i]))
        else:
            invalid_count += 1

    print('Invalid task count:\t', invalid_count)

    if len(roc_list) < y_true_list.shape[1]:
        print('Some target is missing!')
        print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true_list.shape[1]))

    roc_list = np.array(roc_list)
    roc_value = np.mean(roc_list)
    print('recorded: ', roc_value)
    print('\n\n\n')
