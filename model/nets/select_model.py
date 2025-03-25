from ..layers.decoder_layer import Decoder_mlp
from .causal_attention_learning import cal


# 根据 cal_name 返回不同的cal
def cal_model(args, net_params, cline_edge = None):
    if(args.cal_name == 'cal'):
        return cal(args, net_params[args.cal_name])
    else:
        print("cal_name_error!")
    return 0

# 根据 decoder_name 返回不同的decoder层
def decoder_model(args, net_params):
    if(args.decoder_name == 'decoder_mlp'):
        return Decoder_mlp(args, net_params[args.decoder_name])
    else:
        print("decoder_name_error!")
    return 0