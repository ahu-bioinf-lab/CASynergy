import numpy as np
import pandas as pd
import torch
import torch.utils.data
import time
import csv
import random
import deepchem as dc
import copy
import sys
import pickle
import numpy as np
sys.path.append("..")
from utils.utils import drug_feature_extract, get_MACCS, gene_map, read_bed_file, read_csv_header, get_sim_mat
from rdkit import Chem
from torch_geometric import data as DATA
from torch_geometric.data import Batch

# 定义数据库类
class Dataset(torch.utils.data.Dataset):

    def __init__(self, DATASET_NAME, args):
        """
            Loading datasets
        """
        start = time.time()
        print("[I] Loading dataset: %s..." % (DATASET_NAME))
        self.DATASET_NAME = DATASET_NAME
        self.args = args
        drug_fea, drug_smiles_fea, cline_gene_exp_fea, cline_mask, synergy = self.load_data(args, DATASET_NAME)
        synergy_c = self.classifier_label(synergy)  # 根据阈值修改标签为0 1
        # 处理细胞系为图结构，添加上PPI网络为边 格式为 [[exp_fea],[ppi_edge]]
        if(args.using_specific_ppi == True):
            cline_fea = self.get_cline_ppi_edge(args.data_dir + DATASET_NAME)
        else:
            cline_fea = self.get_cline_ppi_edge_common(args.data_dir + DATASET_NAME)
        # 获取ppi网络（单次）
        self.cline_fea = cline_fea                                  # 细胞系特征
        self.cline_mask = cline_mask                                # 细胞系掩码
        self.drug_fea = drug_fea                                    # 分子图特征
        self.drug_smiles_fea = drug_smiles_fea                      # 药物 simles串 特征
        self.synergy = synergy                                      # 药物协同数据
        self.synergy_c = synergy_c                                  # 药物协同数据 分类任务
        self.train_set, self.test_set, self.test_index, self.dataset_all = self.build_data(args) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_drug_ids={10, 20}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_drug_ids={11, 19}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_drug_ids={12, 18}) # 划分数据集训练集、测试集、测试集的序号
        self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_drug_ids={13, 17}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_drug_ids={2,3}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_cline_ids={0,2,4,8}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_cline_ids={1,3,5,9}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_cline_ids={0,2,4,8}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_cline_ids={0,2,4,8}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start, self.test_set_cold_start, self.test_index_cold_start = self.build_data_cold_start(args,fixed_cline_ids={0,2,4,8}) # 划分数据集训练集、测试集、测试集的序号
        # self.train_set_cold_start_spcancer, self.test_set_cold_start_spcancer, self.test_index_cold_start_spcancer = self.build_data_cold_start_spcancer(args,fixed_cline_ids={4})
        self.train_set_cold_start_spcancer, self.test_set_cold_start_spcancer, self.test_index_cold_start_spcancer = self.build_data_cold_start_spcancer(args,fixed_cline_ids={18})
        self.inter_set, self.inter_index = self.get_inter_set(args.cline_num)
        # 划分测试集，数据集，和用来五折交叉验证的数据集
        # self.train_set, self.test_set, self.test_index, self.dataset_all = self.build_data_index(args)
        random.seed(10)
        print('train, test sizes :',len(self.train_set),len(self.test_set))
        print('train_spcancer, test_spcancer sizes :',len(self.train_set_cold_start_spcancer),len(self.test_set_cold_start_spcancer))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def load_data(self, args, dataset):
        cline_fea, drug_fea, drug_smiles_fea, synergy = self.getData(dataset, args.data_dir)
        cline_fea = torch.from_numpy(cline_fea)
        cline_mask = self.get_cline_mask(dataset, args.data_dir)
        return drug_fea, drug_smiles_fea, cline_fea, cline_mask, synergy # drug_smiles_fea 是分子指纹特征

    # 获取特异性PPI
    def get_cline_ppi_edge(self, file_path):
        # 读取 cell_line_gene_expression.csv 文件
        gene_expression_df = pd.read_csv(file_path + '/' + 'cell line_gene_expression.csv', index_col=0)
        # 创建一个蛋白序号映射字典
        protein_mapping = {protein: index for index, protein in enumerate(gene_expression_df.columns)}
        # 更新列名
        gene_expression_df.columns = gene_expression_df.columns.map(protein_mapping)

        ppi_df = pd.read_excel(file_path + '/' + 'cell_line_ppi.xlsx')

        # 使用之前创建的蛋白序号映射字典来映射蛋白序号
        # 使用map方法进行替换
        # 不是str换不了
        ppi_df['protein_a'] = ppi_df['protein_a'].astype('str')
        ppi_df['protein_b'] = ppi_df['protein_b'].astype('str')
        ppi_df['protein_a'] = ppi_df['protein_a'].map(protein_mapping)
        ppi_df['protein_b'] = ppi_df['protein_b'].map(protein_mapping)

        # 删除包含NaN值的行
        ppi_df.dropna(subset=['protein_a', 'protein_b'], inplace=True)

        # 转换回int类型
        ppi_df['protein_a'] = pd.to_numeric(ppi_df['protein_a'], errors='coerce')
        ppi_df['protein_b'] = pd.to_numeric(ppi_df['protein_b'], errors='coerce')
        cell_lines = gene_expression_df.index.drop_duplicates()
        cell_line_data = []

        for cell_line in cell_lines:
            cell_line_ppi = ppi_df[ppi_df['cell_line'] == cell_line]
            cell_line_edges = cell_line_ppi[['protein_a', 'protein_b']].values.tolist()
            # 使用列表解析进行转置
            cell_line_edges = [[row[i] for row in cell_line_edges] for i in range(len(cell_line_edges[0]))]
            # 二维列表拼接子元素构成一维列表
            # cell_line_edges = [item for edge in cell_line_edges for item in edge]
            cell_line_features = gene_expression_df.loc[cell_line].values.tolist()
            cell_line_data.append([np.asarray(cell_line_features), np.asarray(cell_line_edges)])

        return cell_line_data

    # 获取共有的PPI作拓扑结构
    def get_cline_ppi_edge_common(self, file_path):
        # 读取 cell_line_gene_expression.csv 文件
        gene_expression_df = pd.read_csv(file_path + '/' + 'cell line_gene_expression.csv', index_col=0)
        # 创建一个蛋白序号映射字典
        protein_mapping = {protein: index for index, protein in enumerate(gene_expression_df.columns)}
        # 更新列名
        gene_expression_df.columns = gene_expression_df.columns.map(protein_mapping)

        # 读取共有的PPI文件
        ppi_df = pd.read_excel(file_path + '/' + 'protein-protein_network.xlsx')

        # 使用之前创建的蛋白序号映射字典来映射蛋白序号
        # 使用map方法进行替换
        # 不是str换不了
        ppi_df['protein_a'] = ppi_df['protein_a'].astype('str')
        ppi_df['protein_b'] = ppi_df['protein_b'].astype('str')
        ppi_df['protein_a'] = ppi_df['protein_a'].map(protein_mapping)
        ppi_df['protein_b'] = ppi_df['protein_b'].map(protein_mapping)

        # 删除包含NaN值的行
        ppi_df.dropna(subset=['protein_a', 'protein_b'], inplace=True)

        # 转换回int类型
        ppi_df['protein_a'] = pd.to_numeric(ppi_df['protein_a'], errors='coerce')
        ppi_df['protein_b'] = pd.to_numeric(ppi_df['protein_b'], errors='coerce')
        cell_lines = gene_expression_df.index.drop_duplicates()
        cell_line_data = []

        for cell_line in cell_lines:
            cell_line_ppi = ppi_df
            cell_line_edges = cell_line_ppi[['protein_a', 'protein_b']].values.tolist()
            # 使用列表解析进行转置
            cell_line_edges = [[row[i] for row in cell_line_edges] for i in range(len(cell_line_edges[0]))]
            # 二维列表拼接子元素构成一维列表
            cell_line_features = gene_expression_df.loc[cell_line].values.tolist()
            cell_line_data.append([np.asarray(cell_line_features), np.asarray(cell_line_edges)])

        return cell_line_data

    # 分类任务修改标签
    def classifier_label(self, synergy):
        threshold = 0
        for s in synergy:
            s[3] = 1 if s[3] >= threshold else 0
        return synergy

    def build_data(self, args):
        """ 分割数据集为训练集、测试集 """
        # 读取数据
        synergy = self.synergy_c if args.mode == 0 else self.synergy
        drug_fea = self.drug_fea
        cline_fea = self.cline_fea
        cline_mask = self.cline_mask

        # 深拷贝synergy
        synergy_temp = copy.deepcopy(synergy)

        # 修改cline的index
        for i in synergy:
            i[2] -= len(self.drug_fea)
        # 修改cline的index
        for i in synergy_temp:
            i[2] -= len(self.drug_fea)

        # 从index换成对应的特征
        for index, itr in enumerate(synergy_temp):
            # 使用 i,j,k 保存下标
            i = int(itr[0])
            j = int(itr[1])
            k = int(itr[2])
            # 替换为数据
            itr[0] = drug_fea[i]
            itr[1] = drug_fea[j]
            itr[2] = cline_fea[k]
            itr.insert(3, cline_mask[k])
            # 添加index(手动添加协同数据的序号)
            itr.insert(5, index)
        train_size = 0.9
        synergy_temp = pd.DataFrame([i for i in synergy_temp])
        # dataset_all 是没有划分的完整数据集，用来做交叉验证
        dataset_all = np.array(synergy_temp)
        # 划分训练和测试集，9：1的比例
        train_set, test_set = np.split(np.array(synergy_temp.sample(frac=1,random_state=args.seed)),
                                       [int(train_size*len(synergy_temp))])
        train_set = train_set[:,:5]
        test_set, test_index = test_set[:,:5], test_set[:,5]

        return train_set, test_set, test_index, dataset_all[:,:5]


    def build_data_cold_start(self, args, fixed_drug_ids = None, fixed_cline_ids = None):
        """直接根据固定药物序号划分数据集，用于冷启动"""
        # 读取数据
        synergy = self.synergy_c if args.mode == 0 else self.synergy
        drug_fea = self.drug_fea
        cline_fea = self.cline_fea
        cline_mask = self.cline_mask

        # 深拷贝synergy
        synergy_temp = copy.deepcopy(synergy)

        # 将数据转化为DataFrame
        synergy_temp = pd.DataFrame([i for i in synergy_temp])
        
        # dataset_all 是没有划分的完整数据集，用来做交叉验证
        dataset_all = np.array(synergy_temp)
        
        # 初始化训练集和测试集
        train_set = []
        test_set = []

        if fixed_drug_ids != None:
            # 遍历synergy_temp，根据固定的药物序号划分
            for row in synergy_temp.values:
                # 如果药物在fixed_drug_ids中，加入测试集
                if int(row[0]) in fixed_drug_ids or int(row[1]) in fixed_drug_ids:
                    test_set.append(list(row))
                else:
                    train_set.append(list(row))
        elif fixed_cline_ids != None:
            # 遍历synergy_temp，根据固定的药物序号划分
            for row in synergy_temp.values:
                # 如果药物在fixed_drug_ids中，加入测试集
                if int(row[2]) in fixed_cline_ids:
                    test_set.append(list(row))
                else:
                    train_set.append(list(row))

        # 现在开始替换索引为对应的特征（在划分之后进行）
        for index, itr in enumerate(train_set):
            i = int(itr[0])
            j = int(itr[1])
            k = int(itr[2])
            itr[0] = drug_fea[i]
            itr[1] = drug_fea[j]
            itr[2] = cline_fea[k]
            itr.insert(3, cline_mask[k])  # 插入cline_mask
            itr.insert(5, index)  # 插入原始的index
        
        for index, itr in enumerate(test_set):
            i = int(itr[0])
            j = int(itr[1])
            k = int(itr[2])
            itr[0] = drug_fea[i]
            itr[1] = drug_fea[j]
            itr[2] = cline_fea[k]
            itr.insert(3, cline_mask[k])  # 插入cline_mask
            itr.insert(5, index)  # 插入原始的index

        # 确保train_set和test_set的元素一致后，再转换为NumPy数组
        train_set = np.array(train_set, dtype=object)  # 使用dtype=object，确保容纳不同类型的对象
        test_set = np.array(test_set, dtype=object)
        # 分割训练集和测试集（只保留特征数据）
        train_set = train_set[:, :5]
        test_set, test_index = test_set[:, :5], test_set[:, 5]

        print(train_set.shape, test_set.shape)
        return train_set, test_set, test_index

    def build_data_cold_start_spcancer(self, args, fixed_drug_ids = None, fixed_cline_ids = None):
        """直接根据固定药物序号划分数据集，用于冷启动"""
        # 读取数据 
        synergy = self.synergy_c if args.mode == 0 else self.synergy
        drug_fea = self.drug_fea
        cline_fea = self.cline_fea
        cline_mask = self.cline_mask
        # 选择固定的序号
        cancer_types = ['skin','ovary','large intestine','breast','lung','prostate']
        cancer_type = cancer_types[2]

        tissue_cline_path = './data/Synergy/OncologyScreen/tissue_cline.csv'
        # 读取csv文件
        tissue_cline_df = pd.read_csv(tissue_cline_path, index_col=None)
        # 获取Tissue为cancer_type的行序号
        # 过滤 tissue_cline_df，根据 'Tissue' 列进行过滤
        tissue_cline_df_filtered = tissue_cline_df[tissue_cline_df['Tissue'] == cancer_type]
        original_row_numbers = tissue_cline_df_filtered.index
        tissue_cline_index = original_row_numbers.tolist()

        # 深拷贝synergy 
        synergy_temp = copy.deepcopy(synergy)

        # 将数据转化为DataFrame
        synergy_temp = pd.DataFrame([i for i in synergy_temp])
        
        synergy_temp = synergy_temp[synergy_temp[2].isin(tissue_cline_index)]
        # 初始化训练集和测试集
        train_set = []
        test_set = []

        if fixed_drug_ids != None:
            # 遍历synergy_temp，根据固定的药物序号划分
            for row in synergy_temp.values:
                # 如果药物在fixed_drug_ids中，加入测试集
                if int(row[0]) in fixed_drug_ids or int(row[1]) in fixed_drug_ids:
                    test_set.append(list(row))
                else:
                    train_set.append(list(row))

        elif fixed_cline_ids != None:
            # 遍历synergy_temp，根据固定的药物序号划分
            for row in synergy_temp.values:
                # 如果药物在fixed_drug_ids中，加入测试集
                if int(row[2]) in fixed_cline_ids:
                    test_set.append(list(row))
                else:
                    train_set.append(list(row))

        # 现在开始替换索引为对应的特征（在划分之后进行）
        for index, itr in enumerate(train_set):
            i = int(itr[0])
            j = int(itr[1])
            k = int(itr[2])
            itr[0] = drug_fea[i]
            itr[1] = drug_fea[j]
            itr[2] = cline_fea[k]
            itr.insert(3, cline_mask[k])  # 插入cline_mask
            itr.insert(5, index)  # 插入原始的index
        
        for index, itr in enumerate(test_set):
            i = int(itr[0])
            j = int(itr[1])
            k = int(itr[2])
            itr[0] = drug_fea[i]
            itr[1] = drug_fea[j]
            itr[2] = cline_fea[k]
            itr.insert(3, cline_mask[k])  # 插入cline_mask
            itr.insert(5, index)  # 插入原始的index

        # 确保train_set和test_set的元素一致后，再转换为NumPy数组
        train_set = np.array(train_set, dtype=object)  # 使用dtype=object，确保容纳不同类型的对象
        test_set = np.array(test_set, dtype=object)
        # 分割训练集和测试集（只保留特征数据）
        train_set = train_set[:, :5]
        test_set, test_index = test_set[:, :5], test_set[:, 5]

        print(train_set.shape, test_set.shape)
        return train_set, test_set, test_index
    
    # 得到要解释的数据集
    def get_inter_set(self,cline_num):
        synergy = np.array(self.synergy)
        rows_to_delete = []  # 存储待删除行的索引
        rows_inter = []  # 存储选择细胞系的索引
        # 找到第三列值不为0的行的索引
        for idx, row in enumerate(synergy):
            if row[2] != cline_num :     # 取序号为 x 的细胞系
                rows_to_delete.append(idx)
            else:
                rows_inter.append(idx)
            # 选取固定其中一个药物的细胞系
            # if row[2] != cline_num or (row[1] != 0 and row[0] != 0):     # 取序号为 x 的细胞系
            #     rows_to_delete.append(idx)
            # else:
            #     rows_inter.append(idx)

            # if row[3] != 0:     # 取拮抗的的细胞系
            #     rows_to_delete.append(idx)
            # else:
            #     rows_inter.append(idx)

        # 深拷贝synergy
        synergy_temp = copy.deepcopy(synergy)
        # 删除符合条件的行
        specific_cline = np.delete(synergy_temp, rows_to_delete, axis=0)
        inter_synergy = specific_cline.tolist()

        # 读取数据
        drug_fea = self.drug_fea
        cline_fea = self.cline_fea
        cline_mask = self.cline_mask

        for index, itr in enumerate(inter_synergy):
            # 使用 i,j,k 保存下标
            i = int(itr[0])
            j = int(itr[1])
            k = int(itr[2])
            # 替换为数据
            itr[0] = drug_fea[i]
            itr[1] = drug_fea[j]
            itr[2] = cline_fea[k]
            itr.insert(3, cline_mask[k])


        inter_synergy = pd.DataFrame([i for i in inter_synergy])
        inter_synergy = np.array(inter_synergy)
        inter_index = [item for item in rows_inter]
        inter_set = inter_synergy[:,:]
        return inter_set, inter_index

    def process(self, datas):
        data_list = []
        for data in datas:
            features = torch.Tensor(data[0]);        # 药物的分子结构
            edge_index = torch.LongTensor(data[1])   # 药物的拓扑结构
            GCNData = DATA.Data(x=features, edge_index=edge_index)  # DATA.Data类中x接受了药物的分子结构，edge_index接收了药物的拓扑结构，这样的数据可以方便的放在GCN中训练
            data_list.append(GCNData)
        return data_list

    def process_cline_mask(self, datas):
        data_list = []
        for data in datas:
            features = torch.Tensor(data);        # 细胞系特征
            CNNData = DATA.Data(x=features)          # 封装到DATA中
            data_list.append(CNNData)
        return data_list

    # for cline_edge
    def process_cline_edge(self, datas):
        data_list = []
        for data in datas:
            features = torch.Tensor(data[0]);        # 药物的分子结构
            edge_index = torch.Tensor(data[1])                    # 药物的拓扑结构
            GCNData = DATA.Data(x=features, edge_index=edge_index)  # DATA.Data类中x接受了药物的分子结构，edge_index接收了药物的拓扑结构，这样的数据可以方便的放在GCN中训练
            data_list.append(GCNData)
        return data_list

    def collate(self, samples):
        drug_a, drug_b, clines, cline_masks, labels = map(list,zip(*samples))
        labels = torch.tensor(np.array(labels), dtype=torch.float32)
        drug_a = self.process(drug_a)
        drug_b = self.process(drug_b)
        clines = self.process(clines)
        cline_masks = self.process_cline_mask(cline_masks)
        drug_a = self.collate_graph(drug_a)
        drug_b = self.collate_graph(drug_b)
        clines = self.collate_graph(clines)
        cline_masks = self.collate_graph(cline_masks)

        return drug_a, drug_b, clines, cline_masks, labels

    # 获取边信息
    def collate_cline(self, cline):
        cline = self.process_cline_edge(cline)
        cline = self.collate_graph(cline)
        return cline

    # collate
    def collate_graph(args, data_list):
        batchA = Batch.from_data_list([data for data in data_list])
        return batchA

    def getData(self, dataset, file_path):

        drug_smiles_file = file_path + dataset + '/drug_smiles.csv'
        cline_feature_file = file_path + dataset +'/cell line_gene_expression.csv'
        drug_synergy_file = file_path  + dataset +'/drug_synergy.csv'

        drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
        drug_data = pd.DataFrame()
        # 药物分子指纹数据 simles串
        drug_smiles_fea = []
        featurizer = dc.feat.ConvMolFeaturizer()
        if(dataset == "OncologyScreen"):
            for tup in zip(drug['pubchemid'], drug['isosmiles']):
                mol = Chem.MolFromSmiles(tup[1])
                mol_f = featurizer.featurize(mol)
                drug_data[str(tup[0])] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
                drug_smiles_fea.append(get_MACCS(tup[1]))
        else:
            for tup in zip(drug['DrugBank Accession Number'], drug['isosmiles']):
                mol = Chem.MolFromSmiles(tup[1])
                mol_f = featurizer.featurize(mol)
                drug_data[str(tup[0])] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]
                drug_smiles_fea.append(get_MACCS(tup[1]))
        drug_num = len(drug_data.keys())
        d_map = dict(zip(drug_data.keys(), range(drug_num)))
        drug_fea = drug_feature_extract(drug_data)
        # 读取基因表达数据
        gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
        cline_num = len(gene_data.index)
        c_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
        # 基因表达数据转换为array
        cline_fea = np.array(gene_data, dtype='float32')
        # 获取协同数据
        synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)
        synergy = [[d_map[str(row[0])], d_map[str(row[1])], c_map[row[2]], float(row[3])] for index, row in
                synergy_load.iterrows() if (str(row[0]) in drug_data.keys() and str(row[1]) in drug_data.keys() and
                                            str(row[2]) in gene_data.index)]
        return cline_fea, drug_fea, drug_smiles_fea, synergy

    def get_cline_mask(self, dataset, file_path):

        cell_mask_file = file_path + dataset + '/cell_mask.csv'
        cell_mask_list = []
        cell_mask = pd.read_csv(cell_mask_file, sep=',', header=0, index_col=[0])
        for index, row in cell_mask.iterrows():
            values_array = row.iloc[0:].values
            cell_mask_list.append(values_array)

        return cell_mask_list