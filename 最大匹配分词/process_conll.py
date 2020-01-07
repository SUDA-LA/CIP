# -*- coding:utf-8 -*-
import json
fi = open(".\\data\\data.conll", "r", encoding="UTF-8")
fd = open(".\\data\\word.dict", "w", encoding="UTF-8")
fo = open(".\\data\\data.txt", "w", encoding="UTF-8")
fg = open(".\\data\\ground_truth.json", "w", encoding="UTF-8")
sentences = []
sentence = []
words_set = []
count = 0
while True:
    try:
        line = next(fi)
    except StopIteration:
        break
    line_split = line.split('\t')
    if len(line_split) > 1:
        word = line_split[1]
        sentence.append(word)
        if len(word) > 1:
            count += 1
            words_set.append({'word': word})
    else:
        fo.write(''.join(sentence) + "\n")
        sentences.append(sentence)
        sentence = []

fd.write(json.dumps(words_set))
fg.write(json.dumps(sentences))
fi.close()
fd.close()
fo.close()
fg.close()