from Segmenter import Segmenter
import json
from ValUtils import recall
from ValUtils import precision
from ValUtils import f_measure

segmenter = Segmenter(forward=False)
max_word_len = 10

fd = open(".\\data\\data.txt", 'r', encoding="UTF-8")
fg = open(".\\data\\ground_truth.json", 'r', encoding="UTF-8")

lines = fg.readlines()
json_str = ''.join(lines)
gt = json.loads(json_str)
gt_count = len(gt)

for index in range(gt_count):
    line = next(fd)[0:-1]
    seg = segmenter.segment(line, max_word_len)
    print(seg)
    print(gt[index])
    p = precision(seg, gt[index])
    r = recall(seg, gt[index])
    print("precision: %.3f recall: %.3f F-measure: %.3f" % (p, r, f_measure(p, r)))

fd.close()
fg.close()
