import json
import opts
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import os

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR       # 学习率调度器
from model.layers.decoder_layer import Decoder_mlp
from model.nets.gcn_net import gcn
from model.nets.cross_attention_net import cross_attention
from model.nets.cline_extract import cline_fea_extract
from model.nets.drug_extract import drug_fea_extract
from model.nets.select_model import cal_model, decoder_model
from model.nets.predictor import Decoder_no_cross
from model.nets.GCA import ca  # gated cross attention
from model.nets.CAESynergy import CAESynergy

from time import time as time
from torch.utils.data import DataLoader, Subset
from data.data import LoadData
from train.train import train_epoch, eval_epoch
from utils.utils import result_save, result_save_i, test_result_save
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 正常实验分批次训练
def train(args, dataset, out_file, net_params):
    t0 = time()
    # 预处理
    print("[II] preprocess ...")
    # 设置 gpu
    device = torch.device(args.device)

    # define model
    # gcn 处理药物分子图
    model_gcn_drug = gcn(args, net_params['gcn_drug'])

    # 药物和细胞系特征提取模块
    model_drug_extract = drug_fea_extract(args, model_gcn_drug)
    model_cline_extract = cline_fea_extract(args, net_params['gcn_cline'])

    # 使用cross attention融合数据
    ca_d2c_i = ca(args, net_params['ca'])
    model_cline_e_decoder = cross_attention(args, ca_d2c_i)

    # 使用cal解耦细胞系ppi网络
    model_cal = cal_model(args, net_params)

    # 解码层
    decoder_c = Decoder_mlp(args, net_params['decoder_no_cross'])
    decoder_b = Decoder_mlp(args, net_params['decoder_bias'])
    decoder_d = Decoder_mlp(args, net_params['decoder_no_cross'])
    model_Decoder_nocross = Decoder_no_cross(args, decoder_c, decoder_b, decoder_d)

    model_CAESynergy = CAESynergy(args, model_drug_extract, model_cline_extract, model_cline_e_decoder, model_cal, model_Decoder_nocross).to(device)

    # 定义优化器        torch.optim.Adam() or torch.optim.SGD()
    # optimizer = optim.SGD(model_CAESynergy.parameters(), lr = args.learning_rate, weight_decay= args.L2)
    optimizer = optim.Adam(model_CAESynergy.parameters(), lr = args.learning_rate, weight_decay= args.L2)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # Load dataset..
    trainset, testset, dataset_all = dataset.train_set, dataset.test_set, dataset.dataset_all

    # 使用独立验证方式训练
    if(args.Independent_Testing == True):
        # 独立测试集
        train_loader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,drop_last=True, collate_fn=dataset.collate)
        test_loader = DataLoader(testset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

        # 创建保存结果的csv文件，独立验证集
        # 分类
        if args.mode == 0:
            df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
            df_test = pd.DataFrame(columns=['fold', 'loss', 'AUC', 'AUPR', 'F1', 'ACC'])
            df.to_csv(out_file + '/i_train', index=False)
            df_test.to_csv(out_file + '/i_test', index=False)
            best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
        # 回归
        else:
            df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
            df_test = pd.DataFrame(columns=['fold', 'loss', 'rmse', 'r2', 'r'])
            df.to_csv(out_file + '/i_train_r', index=False)
            df_test.to_csv(out_file + '/i_test_r', index=False)
            best_rmse, best_r2, best_r = 1000, 1000, 1000

        # 初始化模型参数
        model_CAESynergy.reset_parameters()

        # 打印预处理使用时间
        print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))

        test_result = []
        # 训练过程
        print("start train ...")
        for epoch in tqdm(range(args.epochs)):
            t0 = time()
            # 在每个epoch开始时更新学习率
            train_result = []
            val_result = []
            att_list = []
            if args.mode == 0:  # 分类
                # 使用训练集和验证集训练与验证
                train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                test_loss, test_auc, test_aupr, test_f1, test_acc, att_list, pre_ls = eval_epoch(model_CAESynergy, test_loader, args, device)
                # 保存结果
                train_result = [epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]
                test_result = [epoch, test_loss, test_auc, test_aupr, test_f1, test_acc]
                # 保存最好的测试集结果
                if test_acc + test_auc> best_acc + best_auc:
                    best_auc =  test_auc
                    best_aupr = test_aupr
                    best_f1 = test_f1
                    best_acc = test_acc
                    torch.save(model_CAESynergy.state_dict(), './model/save_model/best_model_' + args.out_dir_different_params + '.pth')
            else:               # 回归
                train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                test_loss, test_rmse, test_r2, test_r, att_list, pre_ls = eval_epoch(model_CAESynergy, test_loader, args, device)
                train_result = [epoch, train_loss, loss_c, train_rmse, train_r2, train_r]
                test_result = [epoch, test_loss, test_rmse, test_r2, test_r]
                if test_rmse < best_rmse:
                    best_rmse =  test_rmse
                    best_r2 = test_r2
                    best_r = test_r
                    torch.save(model_CAESynergy.state_dict(), './model/save_model/reg_best_model_' + args.out_dir_different_params  + repr(fold) + '.pth')

            # 保存结果
            result_save_i(args, epoch, out_file, train_result, test_result)
            # cline_att_score_save(args.mode, out_file, att_list)
            # testset_result_save(out_file, dataset.synergy, dataset.test_index, pre_ls)
            print("training done! cost time:[{:.2f}s]".format(time()-t0))
        # 保存并输出最好的结果
        if args.mode == 0:
            best_result = ["best_result", best_auc, best_aupr, best_f1, best_acc]
            best_result_data = pd.DataFrame([best_result])
            best_result_data.to_csv(out_file + '/i_test' , mode='a', header=False, index=False)
        else :
            best_result = ["best_result", best_rmse, best_r2, best_r]
            best_result_data = pd.DataFrame([best_result])
            best_result_data.to_csv(out_file + '/i_test_r', mode='a', header=False, index=False)
    else:
        print("cross_val start train ...")
        # 独立测试集
        test_loader = DataLoader(testset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

        # 平均性能
        if args.mode == 0:
            all_auc, all_aupr, all_f1, all_acc = 0, 0, 0, 0
        else :
            all_rmse, all_r2, all_r = 0, 0, 0

        # sklearn中的kfold五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        # kf = KFold(n_splits=5, shuffle=False)
        # indices = np.arange(len(trainset))
        indices = np.arange(len(dataset_all))

        # folds 是生成器对象，用于生成K次迭代的训练集和测试集索引
        folds = list(kf.split(indices))

        for fold, (trainset_index, valset_index) in enumerate(folds):
            trainset_kf = Subset(dataset_all, trainset_index)
            valset_kf = Subset(dataset_all, valset_index)
            # trainset_kf = Subset(trainset, trainset_index)
            # valset_kf = Subset(trainset, valset_index)
            train_loader = DataLoader(trainset_kf,batch_size=args.batch_size,shuffle=True,drop_last=True, collate_fn=dataset.collate)
            val_loader = DataLoader(valset_kf,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

            # 创建保存结果的csv文件，每一折
            # 分类
            if args.mode == 0:
                df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'AUC', 'AUPR', 'F1', 'ACC'])
                df_val = pd.DataFrame(columns=['fold', 'loss', 'AUC', 'AUPR', 'F1', 'ACC'])
                df.to_csv(out_file + '/train_' +repr(fold), index=False)
                df_val.to_csv(out_file + '/val_' + repr(fold), index=False)
                best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
            # 回归
            else:
                df = pd.DataFrame(columns=['epoch', 'loss', 'loss_c', 'rmse', 'r2', 'r'])
                df_val = pd.DataFrame(columns=['fold', 'loss', 'rmse', 'r2', 'r'])
                df.to_csv(out_file + '/train_r_' +repr(fold), index=False)
                df_val.to_csv(out_file + '/val_r_' + repr(fold), index=False)
                best_rmse, best_r2, best_r = 1000, 1000, 1000

            # 初始化模型参数
            model_CAESynergy.reset_parameters()

            # 打印预处理使用时间
            print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))

            test_result = []
            # 训练过程
            print("fold_" + repr(fold) + "train ...")
            print("start train ...")
            for epoch in tqdm(range(args.epochs)):
                t0 = time()
                train_result = []
                val_result = []
                if args.mode == 0:  # 分类
                    # 使用训练集和验证集训练与验证
                    train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_auc, val_aupr, val_f1, val_acc, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                    # 保存结果
                    train_result = [epoch, train_loss, loss_c, train_auc, train_aupr, train_f1, train_acc]
                    val_result = [epoch, val_loss, val_auc, val_aupr, val_f1, val_acc]
                    # 保存最好的验证集结果
                    if val_acc + val_auc > best_acc + best_auc:
                        best_auc =  val_auc
                        best_aupr = val_aupr
                        best_f1 = val_f1
                        best_acc = val_acc
                        print("best_auc:" + str(best_auc) + " | " +"best_aupr:" + str(best_aupr) +" | ""best_f1:" + str(best_f1) +" | ""best_acc:" + str(best_acc) +" | ")
                        torch.save(model_CAESynergy.state_dict(), './model/save_model/best_model_' + repr(fold) + '_' + args.out_dir_different_params + '.pth')
                else:               # 回归
                    train_loss, loss_c, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                    train_result = [epoch, train_loss, loss_c, train_rmse, train_r2, train_r]
                    val_result = [epoch, val_loss, val_rmse, val_r2, val_r]
                    if val_rmse < best_rmse:
                        best_rmse =  val_rmse
                        best_r2 = val_r2
                        best_r = val_r
                        print("best_rmse:" + str(best_rmse) + " | " +"best_r2:" + str(best_r2) +" | ""best_r:" + str(best_r) )
                        torch.save(model_CAESynergy.state_dict(), './model/save_model/reg_best_model_' + repr(fold) + '_' + args.out_dir_different_params + '.pth')

                # 保存结果
                result_save(args, epoch, out_file, fold, train_result, val_result)
                print("training done! cost time:[{:.2f}s]".format(time()-t0))
                # scheduler.step()

            # 保存并输出最好的结果
            if args.mode == 0:
                best_result = ["best_result", best_auc, best_aupr, best_f1, best_acc]
                all_auc += best_auc
                all_aupr += best_aupr
                all_f1 += best_f1
                all_acc += best_acc
                best_result_data = pd.DataFrame([best_result])
                best_result_data.to_csv(out_file + '/val_' + repr(fold), mode='a', header=False, index=False)
                if fold == 4:
                    all_auc, all_aupr, all_f1, all_acc = all_auc/5, all_aupr/5, all_f1/5, all_acc/5
                    all_result = ["avg_result", all_auc, all_aupr, all_f1, all_acc]
                    all_result_data = pd.DataFrame([all_result])
                    all_result_data.to_csv(out_file + '/avg', mode='a', header=False, index=False)
            else :
                best_result = ["best_result", best_rmse, best_r2, best_r]
                all_rmse += best_rmse
                all_r2 += best_r2
                all_r += best_r
                best_result_data = pd.DataFrame([best_result])
                best_result_data.to_csv(out_file + '/val_r_' + repr(fold), mode='a', header=False, index=False)
                if fold == 4:
                    all_rmse, all_r2, all_r = all_rmse/5, all_r2/5, all_r/5
                    all_result = ["avg_result", all_rmse, all_r2, all_r]
                    all_result_data = pd.DataFrame([all_result])
                    all_result_data.to_csv(out_file + '/avg_r', mode='a', header=False, index=False)
            # 加载保存的最好模型
            # model_CAESynergy.load_state_dict(torch.load ('./model/save_model/best_model_' + repr(fold) + '_' + args.out_dir_different_params + '.pth'))

            # if args.mode == 0:  # 分类
            #     test_loss, test_auc, test_aupr, test_f1, test_acc, _, _ = eval_epoch(model_CAESynergy, test_loader, args, device)
            #     test_result = [epoch, test_loss, test_auc, val_aupr, test_f1, test_acc]
            #     test_result_save(args, out_file, fold, test_result)
            # else:
            #     test_loss, test_rmse, test_r2, test_r, _, _ = eval_epoch(model_CAESynergy, test_loader, args, device)
            #     test_result = [epoch, test_loss, test_rmse, test_r2, test_r]
            #     test_result_save(args, out_file, fold, test_result)

def main_params():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]  # 根据不同的数据集读取对应的模型参数

    out_dir_different_params = args.out_dir_different_params

    params_i = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    params_j = [0, 1]
    for i in params_i:
        for j in params_j:
            args.alpha = i
            args.beta = j
            args.out_dir_different_params = out_dir_different_params + '_' + str(args.alpha) + str(args.beta)
            # 结果保存路径
            out_file = args.out_dir + DATASET_NAME + '/' + args.out_dir_different_params
            # 如果不存在文件夹则创建文件夹
            if not os.path.exists(out_file):
                os.mkdir(out_file)
            # 使用指定的方法训练
            if(args.method_name == 'CAESynergy'):
                train(args, dataset, out_file, net_params)
            else:
                print("method_name_error!")

def main_ablation():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]  # 根据不同的数据集读取对应的模型参数

    out_dir_different_params = args.out_dir_different_params
    ablation_param = ['wo_ppi','wo_ca']
    for param in ablation_param:
        if param == 'wo_ppi':
            args.using_specific_ppi = False
            args.ablation_cross_att = False
            # 结果保存路径
            out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params + "_" + param
            # 如果不存在文件夹则创建文件夹
                # 如果不存在文件夹则创建文件夹
            if not os.path.exists(out_file):
                os.mkdir(out_file)
            # 使用指定的方法训练
            if(args.method_name == 'CAESynergy'):
                train(args, dataset, out_file, net_params)
            else:
                print("method_name_error!")
        if param == 'wo_ca':
            args.using_specific_ppi = True
            args.ablation_cross_att = True
            # 结果保存路径
            out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params + "_" + param
            # 如果不存在文件夹则创建文件夹
                # 如果不存在文件夹则创建文件夹
            if not os.path.exists(out_file):
                os.mkdir(out_file)
            # 使用指定的方法训练
            if(args.method_name == 'CAESynergy'):
                train(args, dataset, out_file, net_params)
            else:
                print("method_name_error!")

# 参数敏感性实验 batch_size
def main_bs():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]  # 根据不同的数据集读取对应的模型参数

    out_dir_different_params = args.out_dir_different_params
    bs_param = [16,32,64,128,256]
    for param in bs_param:
        args.batch_size = param
        # 结果保存路径
        out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params + "_bs_" + str(param)
        # 如果不存在文件夹则创建文件夹
            # 如果不存在文件夹则创建文件夹
        if not os.path.exists(out_file):
            os.mkdir(out_file)
        # 使用指定的方法训练
        if(args.method_name == 'CAESynergy'):
            train(args, dataset, out_file, net_params)
        else:
            print("method_name_error!")
        
# 参数敏感性实验 learning_rate
def main_lr():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]  # 根据不同的数据集读取对应的模型参数

    out_dir_different_params = args.out_dir_different_params
    lr_param = [0.01,0.001,0.0001,0.00001,0.000001]
    for param in lr_param:
        args.learning_rate = param
        # 结果保存路径
        out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params + "_lr_" + str(param)
        # 如果不存在文件夹则创建文件夹
            # 如果不存在文件夹则创建文件夹
        if not os.path.exists(out_file):
            os.mkdir(out_file)
        # 使用指定的方法训练
        if(args.method_name == 'CAESynergy'):
            train(args, dataset, out_file, net_params)
        else:
            print("method_name_error!")


def main():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]  # 根据不同的数据集读取对应的模型参数

    out_dir_different_params = args.out_dir_different_params
        # 结果保存路径
    out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params
    # 如果不存在文件夹则创建文件夹
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    # 使用指定的方法训练
    if(args.method_name == 'CAESynergy'):
        train(args, dataset, out_file, net_params)
    else:
        print("method_name_error!")

if __name__ == '__main__':
    # main()
    # main_params()
    # main_ablation()
    # main_bs()
    main_lr()
