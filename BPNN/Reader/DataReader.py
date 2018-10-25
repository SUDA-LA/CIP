class DataReader:
    def __init__(self, path, encoding='UTF-8'):
        fi = open(path, "r", encoding=encoding)
        sentences = []
        sentence = []
        self.tags_reverse = {0: 'START', 1: 'STOP'}
        self.tags = {'START': 0, 'STOP': 1}
        self.words_reverse = {0: '__#O0V?#'}
        self.words = {'__#O0V?#': 0}
        while True:
            try:
                line = next(fi)
            except StopIteration:
                break
            line_split = line.split('\t')
            if len(line_split) > 1:
                word = line_split[1]
                tag = line_split[3]
                part = {'word': word, 'pos': tag}
                sentence.append(part)
                if word not in self.words:
                    w_id = len(self.words)
                    self.words[word] = w_id
                    self.words_reverse[w_id] = word
                if tag not in self.tags:
                    t_id = len(self.tags)
                    self.tags[tag] = t_id
                    self.tags_reverse[t_id] = tag
            else:
                sentences.append(sentence)
                sentence = []

        self.data = sentences
        fi.close()

    def get_seg_data(self, name=True):
        if name:
            return [[word['word'] for word in sentence] for sentence in self.data]
        else:
            return [[self.words[word['word']] for word in sentence] for sentence in self.data]

    def get_pos_data(self, name=True):
        if name:
            return [[word['pos'] for word in sentence] for sentence in self.data]
        else:
            return [[self.tags[word['pos']] for word in sentence] for sentence in self.data]

    def wid2name(self, s):
        return [self.words_reverse[w_id] for w_id in s]

    def name2wid(self, s):
        return [self.words[t] for t in s]

    def tid2name(self, s):
        return [self.tags_reverse[t_id] for t_id in s]

    def name2tid(self, s):
        return [self.tags[t] for t in s]

    def get_data(self, apart=True):
        if apart:
            return self.get_seg_data(), self.get_pos_data()
        else:
            return self.data

    def get_tag_size(self):
        return len(self.tags)

    def get_word_size(self):
        return len(self.words)