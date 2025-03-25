import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import BCELoss,MSELoss,KLDivLoss

from utils.utils import get_mutual, save_features_with_labels_as_csv
from .metrics import metrics_graph, regression_metric

def train_epoch(model_CAESnergy, train_loader, optimizer, args, device):
    model_CAESnergy.train()
    print("[III] training ...")
    # 计算metrics
    true_ls, pre_ls = [], []
    loss_train = 0
    loss_1_train = 0
    # optimizer.zero_grad()   # !!!导致Adam不能用的罪魁祸首
    criterion = BCELoss()
    criterion_r = MSELoss()
    criterion_kl = KLDivLoss(reduction='batchmean')
    # 分类
    if args.mode == 0:
        # druga_se = druga_side_effect
        for batch, (druga, drugb, cline, cline_mask, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score_c, score_d, score_b, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
            # 均匀分布 这里是让预测的概率是 0.5
            # uniform_target = torch.ones_like(score_c, dtype=torch.float).to(device) / 2
            # 现在参考论文“ Shift-Robust Molecular Relational Learning with Causal Substructure ”的另一个做法，
            # 对均匀分布进行随机采样得到一个随机标签 : random_labels
            random_labels = np.random.randint(2, size=args.batch_size)
            # 转换为 tensor 并传到 gpu 上
            random_labels = torch.tensor(random_labels, dtype=torch.float32, requires_grad=True).to(device)
            # 计算不同的loss
            loss_1 = criterion(score_c, label)
            loss_2 = criterion(score_d, label)
            loss_3 = criterion_kl(torch.log(score_b), random_labels)
            # 整合不同的loss , alpha 和 beta 是超参数
            loss = loss_1 + args.alpha * loss_2 + args.beta * loss_3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 指标计算
            loss_train += loss.item()
            loss_1_train += loss_1.item()
            pre_ls += score_c.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
        return loss_train, loss_1_train, auc, aupr, f1, acc
    # 回归
    else:
        for batch, (druga, drugb, cline, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score_c, score_d, score_b, att= model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
            # 现在参考论文“Shift-Robust Molecular Relational Learning with Causal Substructure”的另一个做法，
            # 采用对正态分布进行随机采样 得到 random_labels
            mean = 10  # 均值
            std_dev = 1  # 标准差
            random_samples = np.random.normal(mean, std_dev, args.batch_size)
            # 转换为 tensor 并传到 gpu 上
            random_labels = torch.tensor(random_samples, dtype=torch.float32, requires_grad=True).to(device)
            loss_1 = criterion_r(score_c, label)
            loss_2 = criterion_r(score_d, label)
            loss_3 = criterion_r(score_b, random_labels)
            loss = loss_1 + args.alpha * loss_2 + args.beta * loss_3
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            loss_1_train += loss_1.item()
            pre_ls += score_c.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        rmse, r2, r = regression_metric(pre_ls, true_ls)
        return loss_train.cpu(loss), loss_1_train, rmse, r2, r

def eval_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                loss = criterion(score_c, label)
                pre_ls += score_c.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            return loss_train, auc, aupr, f1, acc, att_ls, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, att_ls, pre_ls

def inter_qa_epoch(model_CAESnergy, eval_loader, args, device, out_file):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    # 创建空列表来记录所有批次的特征
    o_fea_list = []
    c_fea_list = []
    nc_fea_list = []
    labels_list = []
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                # origin_feature causal_feature noncausal_feature
                o_fea, c_fea, nc_fea = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                
                # 将每个批次的特征添加到对应的列表中
                o_fea_list.append(o_fea.cpu().numpy())  # 将张量转换为NumPy数组
                c_fea_list.append(c_fea.cpu().numpy())  # 同上
                nc_fea_list.append(nc_fea.cpu().numpy())  # 同上
                labels_list.append(label.cpu().numpy())  # 保存标签

                # mutual_list = get_mutual(c_fea, nc_fea, o_fea, label, device)

            # 将所有批次的特征合并为一个大的数组
            o_fea_all = np.concatenate(o_fea_list, axis=0)
            c_fea_all = np.concatenate(c_fea_list, axis=0)
            nc_fea_all = np.concatenate(nc_fea_list, axis=0)
            labels_all = np.concatenate(labels_list, axis=0)

            # 保存到CSV文件
            save_features_with_labels_as_csv(c_fea_all, nc_fea_all, o_fea_all, labels_all, out_file, args.cline_num)
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, att_ls, pre_ls

def inter_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            # druga_se = druga_side_effect
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                loss = criterion(score_c, label)
                pre_ls += score_c.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            return loss_train, auc, aupr, f1, acc, att_ls, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                att_ls += att.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, att_ls, pre_ls

# 案例分析实验
def case_study_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            # druga_se = druga_side_effect
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                pre_ls += score_c.cpu().detach().numpy().tolist()
            return loss_train, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score_c, _, _, att = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score_c,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score_c.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            return loss_train, pre_ls

# 消融实验训练
def train_ablation_epoch(model_CAESnergy, train_loader, optimizer, args, device):
    model_CAESnergy.train()
    print("[III] training ...")
    # 计算metrics
    true_ls, pre_ls = [], []
    loss_train = 0
    loss_1_train = 0
    # optimizer.zero_grad()   # !!!导致Adam不能用的罪魁祸首
    criterion = BCELoss()
    criterion_r = MSELoss()
    criterion_kl = KLDivLoss(reduction='batchmean')
    # 分类
    if args.mode == 0:
        # druga_se = druga_side_effect
        for batch, (druga, drugb, cline, cline_mask, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()   # !!!导致Adam不能用的罪魁祸首
            # 指标计算
            loss_train += loss.item()
            pre_ls += score.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
        return loss_train, auc, aupr, f1, acc
    # 回归
    else:
        for batch, (druga, drugb, cline, label) in enumerate(train_loader):
            label = label.to(device)
            # 返回三个预测得分和节点的注意力得分
            score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
            loss = criterion_r(score, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()   # !!!导致Adam不能用的罪魁祸首
            loss_train += loss.item()
            pre_ls += score.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        rmse, r2, r = regression_metric(pre_ls, true_ls)
        return loss_train.cpu(loss), rmse, r2, r

def eval_ablation_epoch(model_CAESnergy, eval_loader, args, device):
    model_CAESnergy.eval()
    # 计算metrics
    true_ls, pre_ls = [], []
    att_ls = []                 # 注意力分数的列表
    loss_train = 0
    criterion = BCELoss()
    criterion_r = MSELoss()
    with torch.no_grad():
        if args.mode == 0:
            for batch, (druga, drugb, cline, cline_mask, label) in enumerate(eval_loader):
                score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device), cline_mask.to(device))
                label = label.to(device)
                loss = criterion(score, label)
                pre_ls += score.cpu().detach().numpy().tolist()
                true_ls += label.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            auc, aupr, f1, acc = metrics_graph(pre_ls, true_ls)
            return loss_train, auc, aupr, f1, acc, pre_ls
        else:
            for batch, (druga, drugb, cline, label) in enumerate(eval_loader):
                score = model_CAESnergy(druga.to(device), drugb.to(device), cline.to(device))
                label = label.to(device)
                loss = criterion_r(score,label)
                true_ls += label.cpu().detach().numpy().tolist()
                pre_ls += score.cpu().detach().numpy().tolist()
                loss_train += loss.item()
            rmse, r2, r = regression_metric(pre_ls, true_ls)
            return loss_train, rmse, r2, r, pre_ls