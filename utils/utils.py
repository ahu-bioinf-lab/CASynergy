import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from .similarity import get_Cosin_Similarity, get_pvalue_matrix
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_classif
import os
# 这两个函数用来计算得到的分子图特征，通过给出的list中的两个ndarray来计算。
# -----molecular_graph_feature
def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]

def drug_feature_extract(drug_data):
    drug_data = pd.DataFrame(drug_data).T
    drug_feat = [[] for _ in range(len(drug_data))]
    for i in range(len(drug_feat)):
        feat_mat, adj_list = drug_data.iloc[i]
        drug_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return drug_feat


# 计算分子指纹，返回值为arr是表示根据smiles计算的分子指纹
def get_MACCS(smiles):
    m = Chem.MolFromSmiles(smiles) # SMILES字符串转换为RDKit中的分子对象，存储在变量m中。
    arr = np.zeros((1,), np.float32)
    fp = MACCSkeys.GenMACCSKeys(m)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# 将给定的神经网络模型nn中所有可重置的参数重置为默认值。
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

# 分类模型的性能指标的，它的输入参数是真实标签值 yt 和模型预测标签值 yp。
# 具体来说，它计算了以下指标：ROC曲线下面积（AUC）、精确度-召回率曲线下面积（AUPR）、F1分数、准确率
def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    # ---f1,acc,recall, specificity, precision
    real_score = np.mat(yt)
    predict_score = np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0]  # , recall[0, 0], specificity[0, 0], precision[0, 0]


# mode = 0 是分类任务、mode = 1 是回归任务
def result_save(args, epoch, out_file, fold, train_result, val_result):
    # 分类
    if args.mode == 0:
        train_result_data = pd.DataFrame([train_result])
        val_result_data = pd.DataFrame([val_result])
        train_result_data.to_csv(out_file + '/train_' +repr(fold), mode='a', header=False, index=False)
        val_result_data.to_csv(out_file + '/val_' + repr(fold), mode='a', header=False, index=False)
    # 回归
    else:
        train_result_data = pd.DataFrame([train_result])
        val_result_data = pd.DataFrame([val_result])
        train_result_data.to_csv(out_file + '/train_r_' +repr(fold), mode='a', header=False, index=False)
        val_result_data.to_csv(out_file + '/val_r_' + repr(fold), mode='a', header=False, index=False)

        # mode = 0 是分类任务、mode = 1 是回归任务
def cold_start_result_save(args, epoch, out_file, fold, result):
    # 分类
    if args.mode == 0:
        train_result_data = pd.DataFrame([result])
        train_result_data.to_csv(out_file + '/cold_start_train_' +repr(fold), mode='a', header=False, index=False)
    # 回归
    else:
        train_result_data = pd.DataFrame([result])
        train_result_data.to_csv(out_file + '/cold_start_train_r_' +repr(fold), mode='a', header=False, index=False)

# mode = 0 是分类任务、mode = 1 是回归任务  i代表独立测试集
def result_save_i(args, epoch, out_file, train_result, test_result):
    # 分类
    if args.mode == 0:
        train_result_data = pd.DataFrame([train_result])
        test_result_data = pd.DataFrame([test_result])
        train_result_data.to_csv(out_file + '/i_train', mode='a', header=False, index=False)
        test_result_data.to_csv(out_file + '/i_test', mode='a', header=False, index=False)
    # 回归
    else:
        train_result_data = pd.DataFrame([train_result])
        test_result_data = pd.DataFrame([test_result])
        train_result_data.to_csv(out_file +  '/i_train', mode='a', header=False, index=False)
        test_result_data.to_csv(out_file +  '/i_train', mode='a', header=False, index=False)

# mode = 0 是分类任务、mode = 1 是回归任务
def test_result_save(args, out_file, fold, test_result):
    # 分类
    if args.mode == 0:
        test_result_data = pd.DataFrame([test_result])
        test_result_data.to_csv(out_file + '/test_' + repr(fold), mode='a', header=False, index=False)
    # 回归
    else:
        test_result_data = pd.DataFrame([test_result])
        test_result_data.to_csv(out_file + '/test_r_' + repr(fold), mode='a', header=False, index=False)

# 预测分数是多少
def testset_result_save(out_file, synergy, test_index, pre_ls):
    df = pd.DataFrame(synergy)
    test_result = []
    # 转换维度[1,n] -> [n,1]
    pre_ls = [[x] for x in pre_ls]
    # 从 synergy 提取药物细胞系信息
    for index in test_index:
        temp_list = df.iloc[index].tolist()
        test_result.append(temp_list)
    # 拼接
    test_result = [row_1 + row_2 for row_1, row_2 in zip(test_result, pre_ls)]
    pd.DataFrame(test_result).to_csv(out_file + '/test_result', index=False)

    return 0

# mode = 0 是分类任务、mode = 1 是回归任务 保存
def cline_att_score_save(args, out_file, att):
    att = np.array(att).reshape(-1, args.cline_dim)   # [cline_num, gene_num]
    # 分类
    if args.mode == 0:
        att_data = pd.DataFrame(att)
        att_data.to_csv(out_file + '/att_data')
    # 回归
    else:
        att_data = pd.DataFrame(att)
        att_data.to_csv(out_file + '/att_data_r')

# 计算回归任务的度量指标
def regression_metric(ytrue, ypred):
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, p = pearsonr(ytrue, ypred)
    return rmse, r2, r

# 映射
def gene_map(list1, list2):
    # 创建字典来保存列表元素和它们的索引
    index_dict = {val: i for i, val in enumerate(list2)}

    # 遍历第二个列表，并在字典中查找对应的索引
    result = [index_dict[val] for val in list1]

    # 交换键和值，以获得索引到值的映射
    result_dict = {i: val for i, val in enumerate(result)}

    return result_dict

# 读取bed文件
def read_bed_file(filename):
    last_column = []  # 存储最后一列的列表
    with open(filename, 'r' , encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                fields = line.split('\t')
                if len(fields) > 3:  # 至少需要四列才有最后一列
                    last_column.append(fields[-1])
    return last_column

# 读取csv文件的第一行
def read_csv_header(filename):
    df = pd.read_csv(filename, header=0)
    df = df.columns.tolist()
    return df[1:]

# 获取表示相似性的函数
def get_sim_mat(drug_fea, cline_fea, device):
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(cline_fea))
    return torch.from_numpy(drug_sim_matrix).type(torch.FloatTensor).to(device), torch.from_numpy(
        cline_sim_matrix).type(torch.FloatTensor).to(device)

# 计算细胞系因果特征，非因果特征，原特征，与标签的互信息
def get_mutual(cline_causal_fea, cline_noncausal_fea, cline_fea, y, device):
    # causal_mutual
    cline_causal_fea = cline_causal_fea.cpu().detach().numpy().tolist()
    cline_noncausal_fea = cline_noncausal_fea.cpu().detach().numpy().tolist()
    cline_fea = cline_fea.cpu().detach().numpy().tolist()
    y = y.cpu().detach().numpy().tolist()

    cm = mutual_info_classif(cline_causal_fea, y, discrete_features = False)
    # noncausal_mutual
    nm = mutual_info_classif(cline_noncausal_fea, y, discrete_features = False)
    # origin_mutual
    om = mutual_info_classif(cline_fea, y, discrete_features = False)
    return [cm, nm, om]

def save_features_with_labels_as_csv(cline_causal_fea, cline_noncausal_fea, cline_fea, y, save_dir, cline_num):
    # 将特征转换为numpy数组
    # cline_causal_fea = cline_causal_fea.cpu().detach().numpy()
    # cline_noncausal_fea = cline_noncausal_fea.cpu().detach().numpy()
    # cline_fea = cline_fea.cpu().detach().numpy()
    # y = y.cpu().detach().numpy()  # 确保标签是numpy数组

    cline_causal_fea = cline_causal_fea
    cline_noncausal_fea = cline_noncausal_fea
    cline_fea = cline_fea
    y = y

    # 定义特征和对应的文件名
    features = {
        "cline_causal_fea": cline_causal_fea,
        "cline_noncausal_fea": cline_noncausal_fea,
        "cline_fea": cline_fea
    }

    # 确保目标文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存每个特征和标签到CSV文件
    for feature_name, feature_data in features.items():
        # 将特征数据和标签合并
        feature_with_labels = pd.DataFrame(feature_data)
        feature_with_labels['label'] = y  # 将标签作为最后一列

        # 保存为CSV文件
        feature_with_labels.to_csv(os.path.join(save_dir, f"{feature_name}_{cline_num}.csv"), index=False)

