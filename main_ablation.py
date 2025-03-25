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
from model.nets.drug_extract import drug_fea_extract
from model.nets.cline_extract import cline_fea_extract
from model.nets.predictor import predictor_ablation
from model.nets.GCA import ca, CrossAttention # gated cross attention
from model.nets.CAESynergy import CAESynergy_Ablation

from time import time as time
from torch.utils.data import DataLoader, Subset
from data.data import LoadData
from train.train import train_epoch, eval_epoch, train_ablation_epoch, eval_ablation_epoch
from utils.utils import result_save, result_save_i
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 消融实验分批次训练
def train_ablation(args, dataset, out_file, net_params):
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
    model_cross_att = cross_attention(args, ca_d2c_i)

    # 解码层
    decoder = Decoder_mlp(args, net_params['decoder_ablation'])
    model_Decoder_nocross = predictor_ablation(args, decoder)

    model_CAESynergy = CAESynergy_Ablation(args, model_drug_extract, model_cline_extract, model_cross_att, model_Decoder_nocross).to(device)

    # 定义优化器        torch.optim.Adam() or torch.optim.SGD()
    # optimizer = optim.SGD(model_CAESynergy.parameters(), lr = args.learning_rate, weight_decay= args.L2)
    optimizer = optim.Adam(model_CAESynergy.parameters(), lr = args.learning_rate, weight_decay= args.L2)

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
            df = pd.DataFrame(columns=['epoch', 'loss', 'AUC', 'AUPR', 'F1', 'ACC'])
            df_test = pd.DataFrame(columns=['fold', 'loss', 'AUC', 'AUPR', 'F1', 'ACC'])
            df.to_csv(out_file + '/i_train', index=False)
            df_test.to_csv(out_file + '/i_test', index=False)
            best_auc, best_aupr, best_f1, best_acc = 0, 0, 0, 0
        # 回归
        else:
            df = pd.DataFrame(columns=['epoch', 'loss', 'rmse', 'r2', 'r'])
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
                train_loss, train_auc, train_aupr, train_f1, train_acc = train_ablation_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                test_loss, test_auc, test_aupr, test_f1, test_acc, pre_ls = eval_ablation_epoch(model_CAESynergy, test_loader, args, device)
                # 保存结果
                train_result = [epoch, train_loss, train_auc, train_aupr, train_f1, train_acc]
                test_result = [epoch, test_loss, test_auc, test_aupr, test_f1, test_acc]
                # 保存最好的测试集结果
                if test_auc > best_auc:
                    best_auc =  test_auc
                    best_aupr = test_aupr
                    best_f1 = test_f1
                    best_acc = test_acc
                    torch.save(model_CAESynergy.state_dict(), './model/save_model/best_model_' + args.out_dir_different_params + '.pth')
            else:               # 回归
                train_loss, train_rmse, train_r2, train_r = train_ablation_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                test_loss, test_rmse, test_r2, test_r, att_list, pre_ls = eval_ablation_epoch(model_CAESynergy, test_loader, args, device)
                train_result = [epoch, train_loss, train_rmse, train_r2, train_r]
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
        # sklearn中的kfold五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        indices = np.arange(len(dataset_all))

        # 平均性能
        if args.mode == 0:
            all_auc, all_aupr, all_f1, all_acc = 0, 0, 0, 0
        else :
            all_rmse, all_r2, all_r = 0, 0, 0

        # folds 是生成器对象，用于生成K次迭代的训练集和测试集索引
        folds = list(kf.split(indices))
        for fold, (trainset_index,valset_index) in enumerate(folds):
            trainset_kf = Subset(dataset_all, trainset_index)
            valset_kf = Subset(dataset_all, valset_index)
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
                    train_loss, train_auc, train_aupr, train_f1, train_acc = train_ablation_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_auc, val_aupr, val_f1, val_acc, _ = eval_ablation_epoch(model_CAESynergy, val_loader, args, device)
                    # 保存结果
                    train_result = [epoch, train_loss, train_auc, train_aupr, train_f1, train_acc]
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
                    train_loss, train_rmse, train_r2, train_r = train_epoch(model_CAESynergy, train_loader, optimizer, args, device)
                    val_loss, val_rmse, val_r2, val_r, _, _ = eval_epoch(model_CAESynergy, val_loader, args, device)
                    train_result = [epoch, train_loss, train_rmse, train_r2, train_r]
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
    out_file = args.out_dir + DATASET_NAME + '/' + out_dir_different_params + '_' + "ablation_cal"
    # 如果不存在文件夹则创建文件夹
    if not os.path.exists(out_file):
        os.mkdir(out_file)
    train_ablation(args, dataset, out_file, net_params)


if __name__ == '__main__':
    main()
