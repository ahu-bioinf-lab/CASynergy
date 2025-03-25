'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-13 16:21:34
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-15 13:39:15
FilePath: /ACHM/opts.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import numpy as np
import random
import torch
from itertools import product

'''
opts.py文件通常是用来处理命令行参数和配置模型超参数的Python模块。
它通常定义了一个命令行参数解析器和一些默认参数值，以便用户可以通过命令行或配置文件修改模型的参数。
这种方式使得模型的使用更加灵活，并且可以更方便地进行超参数调整和实验比较。
'''

def parse_args():
    str2bool = lambda x: x.lower() == "true"
    # 创建一个解析对象  科普parse就是解析的意思
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotion', type=str, default='')
    parser.add_argument('--L2', type=float, default=1e-8)
    parser.add_argument('--dataset_name', type=str, default='OncologyScreen', help="DrugCombDB or OncologyScreen")
    parser.add_argument('--out_dir', type=str, default='./result/')
    parser.add_argument('--config', type=str, default='./config/CASynergy.json')
    parser.add_argument('--data_dir', type=str, default='./data/Synergy/')
    parser.add_argument('--out_dir_different_params', type=str, default='OncologyScreen_att_all')    # 不同的模型参数对于的结果路径
    parser.add_argument('--method_name', type=str, default='CAESynergy', help="CAESynergy or DeepSynergy or GraphSynergy") # 对比试验的方法
    parser.add_argument('--Independent_Testing', type=str2bool, default=False)               # True为独立验证 False为5折交叉验证

    ####################    参数     #######################
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda:0 or cpu")
    parser.add_argument('--cal_name', type=str, default='cal', help="cal")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cline_dim', type=int, default=2865, help="2865 or 7217")    # 种子值为0 效果比较好
    parser.add_argument('--cline_hidden', type=int, default=4096, help="2048 or 4096")    # 种子值为0 效果比较好
    parser.add_argument('--drug_dim', type=int, default=256, help="256 or ")    # 种子值为0 效果比较好
    parser.add_argument('--mode', type=int, default=0, help="0 或 1 代表分类和回归")
    parser.add_argument('--batch_size', type=int, default=32, help="64")
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)    # 种子值为0 效果比较好
    parser.add_argument('--cline_num', type=int, default=5, help="0 - 28 : 28个细胞系")    

    ####################     cal     #######################
    parser.add_argument('--with_random', type=str2bool, default=True)
    parser.add_argument('--without_node_attention', type=str2bool, default=False)
    parser.add_argument('--without_edge_attention', type=str2bool, default=False)
    parser.add_argument('--cat_or_add', type=str, default="add")    # random add 的方式
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--using_specific_ppi', type=bool, default=True, help="True or False") # 是否使用特异性PPI

    ####################     gcn     #######################
    parser.add_argument('--use_GMP', type=bool, default=True)       # gcn中是否使用全局最大池化（global max pooling）

    ####################     decoder_name     #######################
    parser.add_argument('--decoder_name', type=str, default='decoder_mlp', help='decoder_lstm or decoder_mlp')
    ####################     ablation     #######################
    parser.add_argument('--ablation_cross_att',  type=bool, default=False, help='True or False')

    args = parser.parse_args()  # 属性给予实例args
    print_args(args)            # 打印实例args中的各个参数
    setup_seed(args.seed)       # 设置随机种子为args.seed
    return args

# 打印参数
def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

# 设定python标准库、torch库、numpy库的随机种子，保证代码可以重现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
