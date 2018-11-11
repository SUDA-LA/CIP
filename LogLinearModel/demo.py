from OptimizedTagger import Tagger
from DataReader import DataReader

# tagger = Tagger()
dr = DataReader('.\\data\\dev.conll')

s = dr.get_seg_data()
gt = dr.get_pos_data()

index = range(5, 55, 5)
averaged_perceptron = True
for e in index:
    word_count = 0
    tagger = Tagger('.\\model\\check_point_' + str(e) + '.pickle')
    _, _, acc = tagger.evaluate(eval_reader=dr)

    print("Tagging in epoch %d Accuracy: %.5f" % (e, acc))


tagger = Tagger('.\\model\\check_point_finish.pickle')
_, _, acc = tagger.evaluate(eval_reader=dr)
print("Tagging in epoch finish Accuracy: %.5f" % acc)
