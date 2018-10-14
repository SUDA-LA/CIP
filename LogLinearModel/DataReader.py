class DataReader:
    def __init__(self, path, encoding='UTF-8'):
        fi = open(path, "r", encoding=encoding)
        sentences = []
        sentence = []
        while True:
            try:
                line = next(fi)
            except StopIteration:
                break
            line_split = line.split('\t')
            if len(line_split) > 1:
                word = {'word': line_split[1], 'pos': line_split[3]}
                sentence.append(word)
            else:
                sentences.append(sentence)
                sentence = []

        self.data = sentences
        fi.close()

    def get_seg_data(self):
        return [[word['word'] for word in sentence] for sentence in self.data]

    def get_pos_data(self):
        return [[word['pos'] for word in sentence] for sentence in self.data]

    def get_data(self):
        return self.data
