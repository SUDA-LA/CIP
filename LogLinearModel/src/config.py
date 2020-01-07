config = {
    'train_data_file': './data/train.conll',  # 训练集文件
    'dev_data_file': './data/dev.conll',  # 开发集文件
    'test_data_file': None,  # 测试集文件
    'iterator': 100,  # 最大迭代次数
    'batchsize': 50,  # 批次大小
    'shuffle': True,  # 每次迭代是否打乱数据
    'exitor': 10,  # 连续多少个迭代没有提升就退出
    'regulization': True,  # 是否正则化
    'step_opt': True,  # 是否步长优化,设为true步长会逐渐衰减，否则为初始步长不变
    'eta': 0.5,  # 初始步长
    'C': 0.0001  # 正则化系数,regulization为False时无效
}
