from Config import *
from log_linear_model import LogLinearModel
from datetime import datetime

if __name__ == "__main__":
    mode = input("请选择大数据集(b)或者小数据集(s)：")
    start_time = datetime.now()
    if shuffle: print("打乱数据集...")
    print("#" * 10 + "开始训练" + "#" * 10)
    if mode == 's':
        llm = LogLinearModel(train_data_dir, dev_data_dir)
        llm.mini_batch_train(epoch, exitor, random_seed, learning_rate, decay_rate, lmbda, shuffle)
    elif mode == 'b':
        llm = LogLinearModel(train_bigdata_dir, dev_bigdata_dir, test_bigdata_dir)
        llm.mini_batch_train(epoch, exitor, random_seed, learning_rate, decay_rate, lmbda, shuffle)
    end_time = datetime.now()
    print("用时:" + str(end_time - start_time))