from OptimizedTagger import Tagger
from DataReader import DataReader

# tagger = Tagger()
tagger = Tagger('.\\model\\bigdata_model.pickle')
dr = DataReader('.\\bigdata\\dev.conll')

s = dr.get_seg_data()
gt = dr.get_pos_data()
acc = 0
word_count = 0

for i, val in enumerate(s):
    tag = tagger.tag(val)
    acc += len([index for index, v in enumerate(tag) if v == gt[i][index]])
    word_count += len(tag)

print("Tagging Accuracy: %.5f" % (acc / word_count))
