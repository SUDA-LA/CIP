config = {
    'train_data_file': './data/train.conll',  # 训练集文件
    'dev_data_file': './data/dev.conll',  # 开发集文件
    'test_data_file': None,  # 测试集文件
    'averaged': False,  # 是否使用averaged percetron
    'iterator': 100,  # 最大迭代次数
    'exitor': 10,  # 连续多少次迭代没有提升效果就退出
    'shuffle': True  # 每次迭代是否打乱数据
}
