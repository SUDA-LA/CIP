from Segmenter import Segmenter
import json
from ValUtils import metric, scores

segmenter = Segmenter(forward=False)
max_word_len = 10

fd = open(".\\data\\data.txt", 'r', encoding="UTF-8")
fg = open(".\\data\\ground_truth.json", 'r', encoding="UTF-8")

lines = fg.readlines()
json_str = ''.join(lines)
gt = json.loads(json_str)
gt_count = len(gt)

all_right_count, all_s_count, all_g_count = 0, 0, 0

for index in range(gt_count):
    line = next(fd)[0:-1]
    seg = segmenter.segment(line, max_word_len)
    right_count, s_count, g_count = metric(seg, gt[index])
    all_right_count += right_count
    all_s_count += s_count
    all_g_count += g_count

r, p, f = scores(all_right_count, all_s_count, all_g_count)
print("precision: %.3f recall: %.3f F-measure: %.3f" % (p, r, f))

fd.close()
fg.close()
