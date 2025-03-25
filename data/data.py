""" File to load dataset based on user control from main file
"""
from data.dataset import Dataset

# 根据不同数据集返回对应数据集类型的类实例对象，可以使用预定义的类变量和类方法处理数据集。
def LoadData(DATASET_NAME, args):
    """
        This function is called in the main.py file
        returns:
        dataset object
    """
    return Dataset(DATASET_NAME, args)
