import json
import opts
import torch
import warnings
import os

from tqdm import tqdm

from model.layers.decoder_layer import Decoder_mlp
from model.nets.gcn_net import gcn
from model.nets.cross_attention_net import cross_attention
from model.nets.drug_extract import drug_fea_extract
from model.nets.cline_extract import cline_fea_extract
from model.nets.select_model import cal_model, decoder_model
from model.nets.predictor import Decoder_no_cross
from model.nets.GCA import ca  # gated cross attention
from model.nets.CAESynergy import CAESynergy, CAESynergy_qa

from time import time as time
from torch.utils.data import DataLoader
from data.data import LoadData
from train.train import inter_epoch, case_study_epoch,inter_qa_epoch
from utils.utils import cline_att_score_save, testset_result_save
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 计算因果特征，非因果特征，原始特征 与 标签 的互信息
def inter_qa(args, dataset, out_file, net_params):
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

    model_CAESynergy = CAESynergy_qa(args, model_drug_extract, model_cline_extract, model_cline_e_decoder, model_cal, model_Decoder_nocross).to(device)

    # 加载指定模型参数
    model_CAESynergy.load_state_dict(torch.load ("./model/save_model/best_model_0_OncologyScreen_0.30.6.pth", map_location=device))
    # Load dataset..
    interset = dataset.inter_set

    # 独立测试集
    inter_loader = DataLoader(interset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

    # 打印预处理使用时间
    print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))

    # 训练过程
    print("start train ...")
    for epoch in tqdm(range(1)):
        t0 = time()
        if args.mode == 0:  # 分类
            # 使用训练集和验证集训练与验证
            inter_qa_epoch(model_CAESynergy, inter_loader, args, device, out_file)
            # 打印结果
            print(" qa_down ")
        else:               # 回归
            inter_loss, inter_rmse, inter_r2, inter_r, att_list, pre_ls = inter_qa_epoch(model_CAESynergy, inter_loader, args, device)
            print("rmse:" + str(inter_rmse) + " | " +"r2:" + str(inter_r2) +" | ""r:" + str(inter_r) )

        # 保存结果
        print("done! cost time:[{:.2f}s]".format(time()-t0))

# 获取可解释分数
def inter(args, dataset, out_file, net_params):
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

    # 加载指定模型参数
    model_CAESynergy.load_state_dict(torch.load ('./model/save_model/best_model_0_OncologyScreen.pth', map_location='cuda:0'))
    # Load dataset..
    interset = dataset.inter_set

    # 独立测试集
    inter_loader = DataLoader(interset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

    # 打印预处理使用时间
    print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))

    # 训练过程
    print("start train ...")
    for epoch in tqdm(range(1)):
        t0 = time()
        att_list = []
        if args.mode == 0:  # 分类
            # 使用训练集和验证集训练与验证
            inter_loss, inter_auc, inter_aupr, inter_f1, inter_acc, att_list, pre_ls = inter_epoch(model_CAESynergy, inter_loader, args, device)
            # 打印结果
            print("auc:" + str(inter_auc) + " | " +"aupr:" + str(inter_aupr) +" | ""f1:" + str(inter_f1) +" | ""acc:" + str(inter_acc) +" | ")
        else:               # 回归
            inter_loss, inter_rmse, inter_r2, inter_r, att_list, pre_ls = inter_epoch(model_CAESynergy, inter_loader, args, device)
            print("rmse:" + str(inter_rmse) + " | " +"r2:" + str(inter_r2) +" | ""r:" + str(inter_r) )

        # 保存结果
        cline_att_score_save(args, out_file, att_list)
        testset_result_save(out_file, dataset.synergy, dataset.inter_index, pre_ls)
        print("done! cost time:[{:.2f}s]".format(time()-t0))

def case_study(args, dataset, out_file, net_params):
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

    # 加载指定模型参数
    model_CAESynergy.load_state_dict(torch.load ('C:/Users/zelo/Desktop/CAASynergy-240117/model/save_model/' +
                                                 'best_model_3_Oncology_avgpooling_cal_mask.pth'))
    # Load dataset..
    interset = dataset.inter_set

    # 独立测试集
    inter_loader = DataLoader(interset,batch_size=args.batch_size,shuffle=False,drop_last=True, collate_fn=dataset.collate)

    # 打印预处理使用时间
    print("preprocess done! cost time:[{:.2f} s]".format(time()-t0))

    # 训练过程
    print("start train ...")
    for epoch in tqdm(range(1)):
        t0 = time()
        att_list = []
        if args.mode == 0:  # 分类
            # 使用训练集和验证集训练与验证
            inter_loss, pre_ls = case_study_epoch(model_CAESynergy, inter_loader, args, device)

        else:               # 回归
            inter_loss, pre_ls = case_study_epoch(model_CAESynergy, inter_loader, args, device)

        testset_result_save(out_file, dataset.synergy, dataset.inter_index, pre_ls)
        print("done! cost time:[{:.2f}s]".format(time()-t0))

def main():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    DATASET_NAME = args.dataset_name
    print("DATASET_NAME:" , DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args)
    net_params = config[DATASET_NAME]

    # 结果保存路径
    out_file = args.out_dir + DATASET_NAME + '/' + args.out_dir_different_params

    # 如果不存在文件夹则创建文件夹
    if not os.path.exists(out_file):
        os.mkdir(out_file)
   
    inter(args, dataset, out_file, net_params)
    # case_study(args, dataset, out_file, net_params)

def main_qa():
    args = opts.parse_args()
    opts.setup_seed(args.seed)

    # 获取配置文件config
    with open(args.config) as f:
        config = json.load(f)

    # 结果保存路径
    for i in range(29):
        args.cline_num = i
        DATASET_NAME = args.dataset_name
        print("DATASET_NAME:" , DATASET_NAME)
        dataset = LoadData(DATASET_NAME, args)
        net_params = config[DATASET_NAME]
        out_file = args.out_dir + DATASET_NAME + '/' + args.out_dir_different_params + '_' + 'qa'
        # 如果不存在文件夹则创建文件夹
        if not os.path.exists(out_file):
            os.mkdir(out_file)
        inter_qa(args, dataset, out_file, net_params)

if __name__ == '__main__':
    main()
    # main_qa()
