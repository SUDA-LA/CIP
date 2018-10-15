from OptimizedTagger import Tagger
from DataReader import DataReader

# tagger = Tagger()

dr = DataReader('.\\data\\dev.conll')

s = dr.get_seg_data()
gt = dr.get_pos_data()

index = range(5, 15, 5)
for e in index:
    acc = 0
    word_count = 0
    tagger = Tagger('.\\model\\check_point_' + str(e) + '.pickle')
    for i, val in enumerate(s):
        tag = tagger.tag(val)
        acc += len([index for index, v in enumerate(tag) if v == gt[i][index]])
        word_count += len(tag)

    print("Tagging in epoch %d Accuracy: %.5f" % (e, acc / word_count))
