from hmm_model import HMM
from data_loader import DataLoader
import datetime
from Config import *

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    hmm1 = HMM(train_bigdata_dir)
    hmm1.evaluate(DataLoader(dev_bigdata_dir))
    end_time = datetime.datetime.now()
    print('共耗时：' + str(end_time - start_time))

