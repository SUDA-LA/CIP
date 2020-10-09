from Config import *
from global_linear_model import GlobalLinearModel
from datetime import datetime

if __name__ == "__main__":
    mode = input("请选择大数据集(b)或者小数据集(s)：")
    start_time = datetime.now()
    if shuffle: print("打乱数据集...")
    if averaged_percetron: print("使用累加权重...")
    print("#" * 10 + "开始训练" + "#" * 10)
    if mode == 's':
        glm = GlobalLinearModel(train_data_dir, dev_data_dir)
        glm.online_train(epoch, exitor, averaged_percetron, shuffle)
    elif mode == 'b':
        glm = GlobalLinearModel(train_bigdata_dir, dev_bigdata_dir, test_bigdata_dir)
        glm.online_train(epoch, exitor, averaged_percetron, shuffle)
    end_time = datetime.now()
    print("用时:" + str(end_time - start_time))