# -*- coding: utf-8 -*-

DATA = 'data/data.conll'
DICT = 'data/word.dict'
TEXT = 'data/data.txt'
OUTPUT = 'data/data.out'


def create_dict(fdata, fdict):
    max_num, total = 0, 0
    with open(fdata, 'r') as f:
        words = {line.split()[1] for line in f if len(line) > 1}
    total = len(words)
    with open(fdict, 'w') as f:
        for word in words:
            f.write("%s\n" % word)
            if max_num < len(word):
                max_num = len(word)
    return max_num, total


def create_text(fdata, ftext):
    with open(fdata, 'r') as data, open(ftext, 'w') as text:
        for line in data:
            if len(line) > 1:
                word = line.split()[1]
                text.write("%s" % word)
            else:
                text.write("\n")


def max_match(ftext, fdict, fout, max_num=3):
    with open(fdict, 'r') as f:
        words = {line.strip() for line in f}
    with open(ftext, 'r') as text, open(fout, 'w') as out:
        for line in text:
            start = 0
            while start < len(line.strip()):
                end = start + max_num
                while end > start + 1 and line[start:end] not in words:
                    end -= 1
                out.write("%s\n" % line[start:end])
                start = end


def evaluate(fout, fdata):
    with open(fout, 'r') as out, open(fdata, 'r') as data:
        x = [len(line.split()[1]) for line in data if len(line) > 1]
        y = [len(line.strip()) for line in out]
    total_data, total_out = len(x), len(y)

    tp, i, j = 0, 0, 0
    while i < total_data and j < total_out:
        if x[i] == y[j]:
            tp += 1
        else:
            skipx, skipy = x[i], y[j]
            while skipx != skipy:
                if skipx > skipy:
                    j += 1
                    skipy += y[j]
                elif skipx < skipy:
                    i += 1
                    skipx += x[i]
        i += 1
        j += 1
    precision = tp / total_out
    recall = tp / total_data
    f = precision * recall * 2 / (precision + recall)
    return tp, total_out, total_data, precision, recall, f


if __name__ == '__main__':
    print("Create dict of given data")
    max_num, total = create_dict(DATA, DICT)
    print("A total of %d different words, of which the max len is %d" %
          (total, max_num))

    print("Create text of given data")
    create_text(DATA, TEXT)

    print("Segment the words in text")
    max_match(TEXT, DICT, OUTPUT, max_num)

    print("Evaluate the result")
    tp, total_out, total_data, P, R, F = evaluate(OUTPUT, DATA)
    print("Precision: %d / %d = %4f" % (tp, total_out, P))
    print("Recall: %d / %d = %4f" % (tp, total_data, R))
    print("F-value: %4f * %4f * 2 / (%4f + %4f) = %4f" % (P, R, P, R, F))
