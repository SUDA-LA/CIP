config = {
    'train_data_file': './data/train.conll',  # 训练集文件
    'dev_data_file': './data/dev.conll',  # 开发集文件
    'test_data_file': None,  # 测试集文件
    'iterator': 100,  # 最大迭代次数
    'batchsize': 1,  # 批次大小
    'shuffle': False,  # 每次迭代是否打乱数据
    'regulization': False,  # 是否正则化
    'step_opt': False,  # 是否步长优化
    'exitor': 10,  # 连续多少个迭代没有提升就退出
    'eta': 1,  # 初始步长
    'C': 0.00001  # 正则化系数,regulization为False时无效
}
