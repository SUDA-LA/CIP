from Config import *
from bpnn_model import BPNNModel
from datetime import datetime

if __name__ == "__main__":
    start_time = datetime.now()
    if shuffle: print("打乱数据集...")
    print("#" * 10 + "开始训练" + "#" * 10)
    if mode == 's':
        bpnn = BPNNModel(train_data_dir, dev_data_dir, None, layer_sizes, word2vec_dir, word_embed_dim)
        bpnn.mini_batch_train(epoch, exitor, random_seed, batch_size, learning_rate, decay_rate, lmbda, window, shuffle, activation, embedding_freeze)
    elif mode == 'b':
        bpnn = BPNNModel(train_bigdata_dir, dev_bigdata_dir, test_bigdata_dir, layer_sizes, word2vec_dir, word_embed_dim)
        bpnn.mini_batch_train(epoch, exitor, random_seed, batch_size, learning_rate, decay_rate, lmbda, window, shuffle, activation, embedding_freeze)
    end_time = datetime.now()
    print("用时:" + str(end_time - start_time))